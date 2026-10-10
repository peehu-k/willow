import json
import random
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np

import skills
from llm import Generator
from nlp_utils import embed

CONFIDENCE_THRESHOLD = 0.5
# Out-of-scope guard: min cosine similarity to the closest known training pattern
SIMILARITY_THRESHOLD = 0.4
# Above this similarity the input is a near-paraphrase of a training pattern: trust the intent
STRONG_SIMILARITY = 0.8
# Below STRONG, still trust the intent unless the request is open-ended
WEAK_SIMILARITY = 0.6
# Intents where "another one" / "heard it before" should give a new response
REPEATABLE = {"joke", "fun_fact", "riddle", "motivation", "study_tips"}
FALLBACKS = [
    "Sorry, I didn't quite get that. Could you rephrase?",
    "I'm not sure I understand. Try asking me what I can do.",
]
# Factual questions that deserve Wikipedia grounding when sent to the generative model
INFORMATIONAL = re.compile(r"\b(why|how|who|what|when|where|which|explain|describe|difference|"
                           r"invent|discover|history|compare)\b", re.I)


class Session:
    """Per-user conversation memory."""

    def __init__(self):
        self.name = None
        self.last_intent = None
        self.used = {}      # intent -> set of responses already given
        self.history = []   # [(role, text)] recent turns, used as context for the generative model


class ChatBot:
    def __init__(self, model_path="model.npz", intents_path="intents.json"):
        # Weights of the trained IntentNet (PyTorch, see train.py), run here as a NumPy forward pass
        w = np.load(model_path)
        self.tags = [str(t) for t in w["tags"]]
        self.layers = [(w["W1"], w["b1"]), (w["W2"], w["b2"]), (w["W3"], w["b3"])]
        with open(intents_path, encoding="utf-8") as f:
            intents = json.load(f)["intents"]
        self.responses = {i["tag"]: i["responses"] for i in intents}
        self.pattern_vecs = embed([p for i in intents for p in i["patterns"]])
        self.sessions = {}
        self.generator = Generator()
        self.jobs = {}  # stream_id -> pending generation
        self._jobs_lock = threading.Lock()

    def forward(self, x):
        """IntentNet inference: Linear-ReLU-Linear-ReLU-Linear (dropout is inactive at inference)."""
        for i, (W, b) in enumerate(self.layers):
            x = x @ W.T + b
            if i < len(self.layers) - 1:
                x = np.maximum(x, 0)
        return x

    def predict(self, text):
        vec = embed([text])
        logits = self.forward(vec)[0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        top = np.argsort(probs)[::-1][:3]
        top3 = [{"intent": self.tags[i], "p": round(float(probs[i]), 3)} for i in top]
        similarity = float((vec @ self.pattern_vecs.T).max())
        return top3, similarity

    def _pick(self, session, tag):
        """Choose a response for tag, avoiding ones this user has already heard."""
        options = self.responses[tag]
        used = session.used.setdefault(tag, set())
        fresh = [r for r in options if r not in used] or options
        if len(fresh) == len(options):
            used.clear()
        choice = random.choice(fresh)
        used.add(choice)
        return choice

    def reply(self, text, session_id="default", tz=None):
        session = self.sessions.setdefault(session_id, Session())
        result = self._reply(text, session, tz)
        if result["intent"] not in ("more", "unknown"):
            session.last_intent = result["intent"]
        if session.name and result["intent"] in ("greeting", "goodbye", "thanks"):
            result["response"] = result["response"].rstrip("!.") + f", {session.name}!"
        session.history.append(("user", text))
        if not result.get("stream_id"):
            session.history.append(("assistant", result["response"]))
        session.history = session.history[-12:]
        return result

    # ---------- generative fallback (streamed) ----------
    def _generate(self, text, session, base):
        if self.generator.backend == "none":
            # No generative backend configured (definitional questions were already tried on Wikipedia)
            return {**base, "intent": "unknown",
                    "response": "I can't answer open-ended questions like that yet. Try asking me "
                                "what something is, or ask what I can do."}
        notes = skills.wikipedia_notes(text) if INFORMATIONAL.search(text) else None
        messages = self.generator.build_messages(session.history, text, notes)
        stream_id = uuid.uuid4().hex
        with self._jobs_lock:
            # drop jobs nobody collected
            cutoff = time.time() - 300
            for k in [k for k, j in self.jobs.items() if j["created"] < cutoff]:
                self.jobs.pop(k, None)
            self.jobs[stream_id] = {"messages": messages, "session": session, "created": time.time(),
                                    "max_tokens": self.generator.max_tokens_for(text)}
        source = self.generator.label + (" + Wikipedia" if notes else "")
        return {**base, "confidence": None, "intent": "open_question", "source": source,
                "response": "", "stream_id": stream_id}

    def stream(self, stream_id):
        """Yield the generated answer for a pending job, recording it in the session history."""
        with self._jobs_lock:
            job = self.jobs.pop(stream_id, None)
        if job is None:
            yield "Sorry, that answer expired. Please ask again."
            return
        parts = []
        try:
            for chunk in self.generator.stream(job["messages"], job["max_tokens"]):
                parts.append(chunk)
                yield chunk
        except Exception:
            msg = "Sorry, I had trouble thinking about that. Please try again."
            parts.append(msg)
            yield msg
        answer = "".join(parts).strip()
        job["session"].history.append(("assistant", answer))
        job["session"].history = job["session"].history[-12:]

    def _reply(self, text, session, tz):
        # 1. Rule-based skills: name memory, arithmetic, world clock, opinions
        name = skills.extract_name(text)
        if name:
            session.name = name
            return {"intent": "set_name", "confidence": 1.0, "source": "rule",
                    "response": f"Nice to meet you, {name}! I'll remember your name."}
        if any(k in text.lower() for k in ("what is my name", "what's my name", "do you know my name")):
            return {"intent": "get_name", "confidence": 1.0, "source": "rule",
                    "response": f"Your name is {session.name}." if session.name
                    else "You haven't told me your name yet. Say 'my name is ...'."}
        answer = skills.try_math(text)
        if answer:
            return {"intent": "math", "confidence": 1.0, "source": "rule", "response": answer}
        answer = skills.world_time(text)
        if answer:
            return {"intent": "world_time", "confidence": 1.0, "source": "rule", "response": answer}
        topic = skills.opinion_topic(text)
        if topic and not re.search(r"\b(me|my|you|yourself)\b", topic):
            wiki = skills.wikipedia_answer(f"what is {topic}")
            if wiki:
                return {"intent": "opinion", "confidence": 1.0, "source": f"Wikipedia: {wiki['title']}",
                        "url": wiki["url"],
                        "response": f"I don't have personal opinions, but here's something interesting about {topic}: {wiki['answer']}"}

        # 2. Deep learning intent classification
        top3, similarity = self.predict(text)
        tag, conf = top3[0]["intent"], top3[0]["p"]
        base = {"confidence": conf, "top3": top3, "source": "neural network"}
        understood = conf >= CONFIDENCE_THRESHOLD and similarity >= SIMILARITY_THRESHOLD
        strong = understood and similarity >= STRONG_SIMILARITY
        open_ended = skills.is_open_ended(text)

        # 3. Knowledge questions the classifier doesn't cover -> Wikipedia.
        # Real intent questions ("what is your name") sit >= 0.93 from a training pattern;
        # look-ups that merely share a word ("what is an LLM" vs "are you an LLM") sit <= 0.75.
        about_us = re.search(r"\b(you|your|yours|yourself|my|me|time|date|today|weather)\b", text.lower())
        if not strong and not open_ended and skills.looks_like_question(text) and (
                not understood or conf < 0.85 or (similarity < 0.85 and not about_us)):
            wiki = skills.wikipedia_answer(text)
            if wiki:
                return {**base, "confidence": None, "intent": "knowledge", "source": f"Wikipedia: {wiki['title']}",
                        "response": wiki["answer"], "url": wiki["url"]}

        # 4. Open-ended or unfamiliar requests -> generative model
        if not strong and (open_ended or not understood or similarity < WEAK_SIMILARITY):
            if (not understood and similarity < 0.25 and len(text.split()) <= 3
                    and not open_ended and not skills.looks_like_question(text)):
                return {**base, "intent": "unknown", "response": random.choice(FALLBACKS)}
            return self._generate(text, session, base)

        # 5. Context: "another one" / "I've heard that" continues the previous topic
        if tag == "repeat":
            last = next((t for r, t in reversed(session.history) if r == "assistant"), None)
            return {**base, "intent": "repeat",
                    "response": f"Sure, I said: {last}" if last else "I haven't said anything yet!"}
        if tag == "more":
            prev = session.last_intent if session.last_intent in REPEATABLE else "joke"
            return {**base, "intent": "more", "response": f"Here's another one: {self._pick(session, prev)}"}

        now = datetime.now(timezone.utc)
        try:
            now = now.astimezone(ZoneInfo(tz)) if tz else now.astimezone()
        except Exception:
            now = now.astimezone()
        response = self._pick(session, tag).format(
            time=f"It's {now.strftime('%I:%M %p').lstrip('0')}.",
            date=f"Today is {now.strftime('%A, %d %B %Y')}.",
        )
        return {**base, "intent": tag, "response": response}

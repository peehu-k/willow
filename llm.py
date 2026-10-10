"""Generative fallback for open-ended requests the intent classifier doesn't cover.

Backends, chosen from environment variables:
  * Groq (GROQ_API_KEY)        - free tier, hosted Llama models, very fast.
  * Claude (ANTHROPIC_API_KEY) - highest quality.
Without a key the chatbot answers open-ended questions from Wikipedia instead (see chatbot.py).
Both backends stream text so the UI can show the answer as it is written.
"""
import json
import os
import sys
import urllib.error
import urllib.request

SYSTEM_PROMPT = (
    "You are Willow, a calm, warm and friendly voice assistant. Your replies are shown in a chat "
    "and read aloud, so write plain text only: no markdown, no asterisks, no hashtags, no tables. "
    "Keep answers short and clear (2 to 5 sentences) unless the user asks for a plan, list, recipe "
    "or itinerary; then use short lines such as 'Day 1: ...'. If reference notes are given, use them "
    "for facts and ignore notes that are not relevant. If you are not sure about a fact, say so "
    "honestly. You have no live data such as news, weather or prices."
)

CLAUDE_MODEL = os.environ.get("WILLOW_CLAUDE_MODEL", "claude-opus-5-5")
ERROR_REPLY = "Sorry, I couldn't reach my thinking service just now. Please try again in a moment."


def _clean(text):
    """Strip markdown symbols models sometimes emit."""
    return text.replace("**", "").replace("__", "").replace("###", "").replace("##", "")


class Generator:
    def __init__(self):
        if os.environ.get("ANTHROPIC_API_KEY"):
            self.backend = "claude"
        elif os.environ.get("GROQ_API_KEY"):
            self.backend = "groq"
        else:
            self.backend = "none"
        self._client = None

    @property
    def label(self):
        return {"claude": "Claude", "groq": "Groq", "none": "none"}[self.backend]

    def warm_up(self):
        pass

    def build_messages(self, history, question, notes=None):
        msgs = [{"role": r, "content": t} for r, t in history[-6:]]
        content = question
        if notes:
            content = f"Reference notes (may or may not be relevant):\n{notes}\n\nUser: {question}"
        msgs.append({"role": "user", "content": content})
        # Chat APIs need the conversation to start with a user turn
        while msgs and msgs[0]["role"] != "user":
            msgs.pop(0)
        return msgs

    def max_tokens_for(self, question):
        return 1500

    def stream(self, messages, max_tokens):
        """Yield text chunks."""
        if self.backend == "claude":
            yield from self._stream_claude(messages, max_tokens)
        else:
            yield from self._stream_groq(messages, max_tokens)

    # ---------- Groq ----------
    _GROQ_PREFERRED = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b",
                       "llama-3.1-8b-instant"]
    _GROQ_SKIP = ("whisper", "guard", "tts", "playai", "orpheus", "safeguard", "embed", "vision")

    def _groq_headers(self):
        key = os.environ["GROQ_API_KEY"].strip().strip(chr(34)).strip(chr(39))
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                "User-Agent": "WillowVoiceBot/1.0"}

    def _groq_pick_model(self, refresh=False):
        """Ask Groq which models this key can use and choose the best chat model.
        Model names change over time, so nothing is hard-coded unless GROQ_MODEL is set."""
        if os.environ.get("GROQ_MODEL"):
            return os.environ["GROQ_MODEL"]
        if getattr(self, "_groq_model", None) and not refresh:
            return self._groq_model
        try:
            req = urllib.request.Request("https://api.groq.com/openai/v1/models", headers=self._groq_headers())
            with urllib.request.urlopen(req, timeout=15) as resp:
                ids = [m["id"] for m in json.load(resp)["data"] if m.get("active", True)]
        except Exception as e:
            print(f"[groq] could not list models: {e!r}", file=sys.stderr, flush=True)
            return self._GROQ_PREFERRED[0]
        chosen = next((m for m in self._GROQ_PREFERRED if m in ids), None) or next(
            (m for m in ids if not any(s in m.lower() for s in self._GROQ_SKIP)), self._GROQ_PREFERRED[0])
        print(f"[groq] using model {chosen} (available: {len(ids)})", file=sys.stderr, flush=True)
        self._groq_model = chosen
        return chosen

    def _stream_groq(self, messages, max_tokens):
        """Groq's hosted models (free tier) via its OpenAI-style streaming endpoint."""
        produced = False
        for attempt in range(2):
            model = self._groq_pick_model(refresh=attempt > 0)
            payload = {
                "model": model,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
                "max_tokens": max_tokens, "temperature": 0.5, "stream": True,
            }
            if "gpt-oss" in model:
                payload["reasoning_effort"] = "low"  # these models think first; keep answers quick
            body = json.dumps(payload).encode()
            req = urllib.request.Request("https://api.groq.com/openai/v1/chat/completions", data=body,
                                         method="POST", headers=self._groq_headers())
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    for raw in resp:
                        line = raw.decode("utf-8").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        choices = json.loads(data).get("choices") or [{}]
                        delta = choices[0].get("delta", {}).get("content")
                        if delta:
                            produced = True
                            yield _clean(delta)
                return
            except urllib.error.HTTPError as e:
                # Shows in the hosting logs (the key is never printed)
                print(f"[groq] HTTP {e.code} with model {model}: {e.read()[:300]!r}", file=sys.stderr, flush=True)
                if e.code in (400, 404) and not produced and attempt == 0:
                    continue  # model unavailable: re-list models and retry once
                break
            except Exception as e:
                print(f"[groq] error: {e!r}", file=sys.stderr, flush=True)
                break
        if not produced:
            yield ERROR_REPLY

    def _stream_claude(self, messages, max_tokens):
        import anthropic
        if self._client is None:
            self._client = anthropic.Anthropic()
        produced = False
        try:
            with self._client.beta.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=messages,
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"],
                extra_body={"fallbacks": "default"},
            ) as stream:
                for text in stream.text_stream:
                    produced = True
                    yield _clean(text)
                final = stream.get_final_message()
            if final.stop_reason == "refusal":
                yield "Sorry, I can't help with that one."
        except anthropic.APIError as e:
            print(f"[claude] error: {e!r}", file=sys.stderr, flush=True)
            if not produced:
                yield ERROR_REPLY

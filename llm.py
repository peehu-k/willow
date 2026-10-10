"""Generative fallback for open-ended requests the intent classifier doesn't cover.

Backends:
  * Claude (if ANTHROPIC_API_KEY is set) - fast, high-quality answers.
  * Qwen2.5-0.5B-Instruct running locally on CPU - free, no key needed.
Both stream text so the UI can show the answer as it is written.
"""
import os
import re
import threading

SYSTEM_PROMPT = (
    "You are Willow, a calm, warm and friendly voice assistant. Your replies are shown in a chat "
    "and read aloud, so write plain text only: no markdown, no asterisks, no hashtags, no tables. "
    "Keep answers short and clear (2 to 5 sentences) unless the user asks for a plan, list, recipe "
    "or itinerary; then use short lines such as 'Day 1: ...'. If reference notes are given, use them "
    "for facts and ignore notes that are not relevant. If you are not sure about a fact, say so "
    "honestly. You have no live data such as news, weather or prices."
)

LONG_TASK = re.compile(r"\b(itinerary|plan|schedule|list|steps|recipe|essay|story|poem|routine|"
                       r"outline|guide|ideas|tips|compare|comparison|write)\b", re.I)

CLAUDE_MODEL = os.environ.get("WILLOW_CLAUDE_MODEL", "claude-opus-5-5")
LOCAL_MODEL = os.environ.get("WILLOW_LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")


def _clean(text):
    """Strip markdown symbols the small model sometimes emits."""
    return text.replace("**", "").replace("__", "").replace("###", "").replace("##", "")


class Generator:
    def __init__(self):
        if os.environ.get("ANTHROPIC_API_KEY"):
            self.backend = "claude"
        elif os.environ.get("GROQ_API_KEY"):
            self.backend = "groq"
        else:
            self.backend = "local"
        self._lock = threading.Lock()  # one local generation at a time (CPU bound)
        self._model = self._tok = self._client = None

    @property
    def label(self):
        return {"claude": "Claude", "groq": "Llama (Groq)", "local": "Qwen2.5 (local)"}[self.backend]

    def warm_up(self):
        if self.backend == "local":
            self._load_local()

    # ---------- message building ----------
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
        if self.backend != "local":
            return 1500
        return 320 if LONG_TASK.search(question) else 170

    def local_max_tokens(self, max_tokens):
        return min(max_tokens, 320)

    # ---------- streaming ----------
    def stream(self, messages, max_tokens):
        """Yield text chunks."""
        if self.backend == "claude":
            yield from self._stream_claude(messages, max_tokens)
        elif self.backend == "groq":
            yield from self._stream_groq(messages, max_tokens)
        else:
            yield from self._stream_local(messages, max_tokens)

    def _stream_groq(self, messages, max_tokens):
        """Groq's hosted Llama models (free tier) via its OpenAI-style streaming endpoint."""
        import json
        import urllib.request
        body = json.dumps({
            "model": os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + messages,
            "max_tokens": max_tokens, "temperature": 0.5, "stream": True,
        }).encode()
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions", data=body, method="POST",
            headers={"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}",
                     "Content-Type": "application/json", "User-Agent": "WillowVoiceBot/1.0"})
        produced = False
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                for raw in resp:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    delta = json.loads(data)["choices"][0].get("delta", {}).get("content")
                    if delta:
                        produced = True
                        yield _clean(delta)
        except Exception:
            if not produced:  # API unavailable: answer with the local model instead
                yield from self._stream_local(messages, self.local_max_tokens(max_tokens))

    def _stream_claude(self, messages, max_tokens):
        import anthropic
        if self._client is None:
            self._client = anthropic.Anthropic()
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
                    yield _clean(text)
                final = stream.get_final_message()
            if final.stop_reason == "refusal":
                yield "Sorry, I can't help with that one."
        except anthropic.APIError:
            # Network/API trouble: fall back to the local model for this answer
            yield from self._stream_local(messages, self.local_max_tokens(max_tokens))

    def _load_local(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            torch.set_num_threads(max(1, os.cpu_count() or 1))
            self._tok = AutoTokenizer.from_pretrained(LOCAL_MODEL)
            self._model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL, dtype=torch.float32)
            self._model.eval()

    def _stream_local(self, messages, max_tokens):
        import torch
        from transformers import TextIteratorStreamer
        self._load_local()
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        enc = self._tok.apply_chat_template(msgs, add_generation_prompt=True,
                                            return_tensors="pt", return_dict=True)
        streamer = TextIteratorStreamer(self._tok, skip_prompt=True, skip_special_tokens=True)

        def run():
            with torch.no_grad():
                self._model.generate(**enc, streamer=streamer, max_new_tokens=max_tokens,
                                     do_sample=False, repetition_penalty=1.1)

        with self._lock:
            worker = threading.Thread(target=run, daemon=True)
            worker.start()
            for text in streamer:
                yield _clean(text)
            worker.join()

# Willow — Voice-Enabled Chatbot (Speech Recognition + Deep Learning)

## Pipeline
```
Mic ─► Speech Recognition ─► text ─┬─► rule skills (maths, world clock, name memory)
                                   ├─► MiniLM encoder ─► IntentNet (deep neural network) ─► intent ─► response
                                   ├─► Wikipedia (definitions: "what is / who is …")
                                   └─► Llama 3.3 70B via Groq (open-ended: "why…", "plan…", "write…")
                                          └─ grounded with Wikipedia notes, streamed to the UI
```

1. **Speech recognition**: the browser Web Speech API (Chrome, Edge, Safari), with a live transcript.
   On other browsers the page records WAV audio and the server transcribes it with the Python
   `SpeechRecognition` library.
2. **Deep learning intent classification** (the core model)
   - `all-MiniLM-L6-v2`, a pretrained 6-layer transformer, turns text into a 384-d sentence embedding.
     It runs with ONNX Runtime, which gives identical embeddings to PyTorch at a fraction of the memory.
   - `IntentNet`, a feed-forward deep neural network (384 → 256 → 128 → 44, ReLU, dropout), is trained in
     PyTorch on `intents.json` (`train.py`). The trained weights (`model.npz`) are served with a NumPy
     forward pass, so the live app needs no PyTorch.
   - Similarity to the nearest training sentence decides whether to trust the intent or hand the question
     to the knowledge or generative layers.
3. **Knowledge layer**: definitional questions are answered from Wikipedia, with a "Read more" link.
4. **Generative layer**: open-ended requests (itineraries, recipes, explanations, advice, follow-ups) go to
   Llama 3.3 70B on Groq's free tier, using recent conversation turns and Wikipedia notes for facts.
   Answers stream into the chat. Set `GROQ_API_KEY` (free) or `ANTHROPIC_API_KEY` (Claude) to enable it.
5. **Extra skills**: per-chat memory ("another one", "repeat that", your name), maths, world clock,
   timezone-aware local time, chat history, and a female read-aloud voice with pause/resume.

## Results
- 44 intents, 344 training sentences
- Training accuracy: 100%; **accuracy on unseen paraphrases: ~94%**
- Server memory: ~200 MB (fits free hosting)

## Run locally
```bash
pip install -r requirements.txt
python setup_models.py   # downloads the MiniLM encoder (ONNX)
python app.py            # http://localhost:7860
```
To retrain the intent network: `pip install torch` and run `python train.py`.

## Deploy (Render, free)
The included `render.yaml` sets everything up: Render → New → Blueprint → pick this repository →
enter your `GROQ_API_KEY` → Apply.

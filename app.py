import io
import os

import speech_recognition as sr
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

from chatbot import ChatBot

app = Flask(__name__)
bot = ChatBot()
recognizer = sr.Recognizer()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    text = data.get("message", "").strip()
    if not text:
        return jsonify({"error": "Empty message"}), 400
    return jsonify({"recognized": text, **bot.reply(text, data.get("session_id", "default"), data.get("tz"))})


@app.route("/stream/<stream_id>")
def stream(stream_id):
    """Streams a generated answer as plain text chunks."""
    return Response(stream_with_context(bot.stream(stream_id)), mimetype="text/plain",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/transcribe", methods=["POST"])
def transcribe():
    """Server-side speech recognition fallback: accepts a WAV recording."""
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"error": "No audio uploaded"}), 400
    try:
        with sr.AudioFile(io.BytesIO(audio_file.read())) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand the audio. Please try again."}), 422
    except sr.RequestError as e:
        return jsonify({"error": f"Speech service unavailable: {e}"}), 503
    except Exception as e:
        return jsonify({"error": f"Invalid audio: {e}"}), 400
    return jsonify({"recognized": text, **bot.reply(
        text, request.form.get("session_id", "default"), request.form.get("tz"))})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 7860)), debug=False)

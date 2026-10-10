// Willow front end: chat UI, persistent conversation history, voice input.
const $ = (id) => document.getElementById(id);
const messagesEl = $("messages");
const historyEl = $("history");
const statusEl = $("status");
const micBtn = $("mic");
const textInput = $("text");
const ttsBtn = $("tts");
const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;

const DEFAULT_STATUS = "Click to speak or start talking...";
const STORE_KEY = "willow.conversations.v1";
const ACTIVE_KEY = "willow.active.v1";
const TTS_KEY = "willow.tts.v1";

// ---------- Persistence (browser localStorage, failure-tolerant) ----------
let storageOk = true;

function readStore(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    storageOk = false;
    return fallback;
  }
}

function writeStore(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    if (storageOk) setStatus("History couldn't be saved on this device.", true);
    storageOk = false;
    return false;
  }
}

let conversations = readStore(STORE_KEY, []);
if (!Array.isArray(conversations)) conversations = [];
conversations = conversations.filter((c) => c && c.id && Array.isArray(c.messages) && c.messages.length);
let draft = null; // a new conversation that has no messages yet (not stored)
let activeId = readStore(ACTIVE_KEY, null);

const saveConversations = () => writeStore(STORE_KEY, conversations);
const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
const findConv = (id) => conversations.find((c) => c.id === id) || (draft && draft.id === id ? draft : null);
const activeConv = () => findConv(activeId);

function startNewConversation() {
  draft = { id: uid(), title: "", createdAt: Date.now(), updatedAt: Date.now(), messages: [] };
  activeId = draft.id;
  writeStore(ACTIVE_KEY, activeId);
}

function openConversation(id) {
  if (!findConv(id)) return;
  stopSpeaking();
  activeId = id;
  writeStore(ACTIVE_KEY, activeId);
  renderAll();
}

function deriveTitle(text) {
  let t = text.trim().replace(/[?.!]+$/, "");
  const filler = /^(?:(?:hello|hi|hey|ok|okay|so|and|um|uh|please|willow|can you|could you|would you|will you|tell me|i want to know|do you know)[\s,]+)+/i;
  const stripped = t.replace(filler, "");
  if (stripped.length >= 3) t = stripped;
  t = t.split(/\s+/).slice(0, 7).join(" ");
  return t.charAt(0).toUpperCase() + t.slice(1) || "New conversation";
}

function addMessage(conv, msg) {
  conv.messages.push(msg);
  conv.updatedAt = msg.ts;
  if (conv === draft) {
    conv.title = deriveTitle(msg.text);
    conversations.unshift(conv);
    draft = null;
  }
  saveConversations();
  renderHistory();
  if (conv.id === activeId) {
    if (messagesEl.querySelector(".empty-state")) messagesEl.innerHTML = "";
    messagesEl.appendChild(bubble(msg));
    scrollToBottom();
  }
}

// ---------- Formatting ----------
const timeOf = (ts) => new Date(ts).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });

function relativeDate(ts) {
  const d = new Date(ts);
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const day = new Date(d); day.setHours(0, 0, 0, 0);
  const diff = Math.round((today - day) / 86400000);
  if (diff === 0) return `Today, ${timeOf(ts)}`;
  if (diff === 1) return `Yesterday, ${timeOf(ts)}`;
  const opts = { month: "short", day: "numeric" };
  if (d.getFullYear() !== today.getFullYear()) opts.year = "numeric";
  return `${d.toLocaleDateString([], opts)}, ${timeOf(ts)}`;
}

const icon = (name) => `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;

// ---------- Rendering ----------
function renderHistory() {
  historyEl.innerHTML = "";
  const sorted = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt);
  if (!sorted.length) {
    historyEl.innerHTML = `<div class="history-empty">Your conversations will appear here.</div>`;
    return;
  }
  for (const c of sorted) {
    const btn = document.createElement("button");
    btn.className = "history-item" + (c.id === activeId ? " active" : "");
    btn.innerHTML = `${icon("chat")}<div class="h-text"><div class="h-title"></div><div class="h-time"></div></div>`;
    btn.querySelector(".h-title").textContent = c.title;
    btn.querySelector(".h-time").textContent = relativeDate(c.updatedAt);
    btn.title = c.title;
    if (c.id === activeId) btn.setAttribute("aria-current", "true");
    btn.onclick = () => openConversation(c.id);
    historyEl.appendChild(btn);
  }
}

function bubble(msg) {
  const row = document.createElement("div");
  row.className = "msg " + msg.role;
  const b = document.createElement("div");
  b.className = "bubble";
  const text = document.createElement("span");
  text.textContent = msg.text;
  b.appendChild(text);
  if (msg.url) {
    const a = document.createElement("a");
    a.href = msg.url; a.target = "_blank"; a.rel = "noopener"; a.textContent = "Read more";
    b.append(" ", a);
  }
  const time = document.createElement("div");
  time.className = "time";
  time.textContent = timeOf(msg.ts);
  b.appendChild(time);
  const avatar = document.createElement("div");
  avatar.className = "avatar " + msg.role;
  avatar.innerHTML = icon(msg.role === "bot" ? "leaf" : "user");
  if (msg.role === "bot") row.append(avatar, b); else row.append(b, avatar);
  return row;
}

function renderMessages() {
  const conv = activeConv();
  messagesEl.innerHTML = "";
  if (!conv || !conv.messages.length) {
    messagesEl.innerHTML = `<div class="empty-state">${icon("leaf")}
      <div class="big">What’s on your mind?</div>
      <div>Tap the microphone and speak, or type a message below.</div></div>`;
    return;
  }
  conv.messages.forEach((m) => messagesEl.appendChild(bubble(m)));
  scrollToBottom(true);
}

function renderInsights() {
  const conv = activeConv();
  const msgs = conv ? conv.messages : [];
  const lastUser = [...msgs].reverse().find((m) => m.role === "user");
  const lastBot = [...msgs].reverse().find((m) => m.role === "bot");
  showRecognized(lastUser ? lastUser.text : null);
  showIntent(lastBot);
}

function showRecognized(text, interim = false) {
  const el = $("recognized");
  el.className = "recognized" + (text ? (interim ? " interim" : "") : " empty");
  el.textContent = text || "Your words will appear here.";
}

function showIntent(msg) {
  const intentEl = $("intent");
  const has = msg && msg.intent;
  intentEl.textContent = has ? msg.intent.replace(/_/g, " ") : "—";
  intentEl.className = "pill" + (has ? "" : " muted");
  const conf = has && typeof msg.confidence === "number" ? msg.confidence : null;
  $("confidence").textContent = conf === null ? "—" : `${(conf * 100).toFixed(1)}%`;
  $("bar-fill").style.width = conf === null ? "0" : `${Math.round(conf * 100)}%`;
  intentEl.title = msg && msg.source ? `Source: ${msg.source}` : "";
}

function renderAll() {
  renderHistory();
  renderMessages();
  renderInsights();
}

function scrollToBottom(force = false) {
  const nearBottom = messagesEl.scrollHeight - messagesEl.scrollTop - messagesEl.clientHeight < 160;
  if (force || nearBottom) messagesEl.scrollTop = messagesEl.scrollHeight;
}

let typingEl = null;
function showTyping(on) {
  if (typingEl) { typingEl.remove(); typingEl = null; }
  if (!on) return;
  if (messagesEl.querySelector(".empty-state")) messagesEl.innerHTML = "";
  typingEl = bubble({ role: "bot", text: "", ts: Date.now() });
  typingEl.classList.add("typing");
  typingEl.querySelector(".bubble").innerHTML = `<span class="dots"><span>•</span><span>•</span><span>•</span></span>`;
  messagesEl.appendChild(typingEl);
  scrollToBottom(true);
}

function setStatus(text, alert = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("alert", alert);
}

// ---------- Talking to the backend ----------
function handleReply(conv, data, source) {
  showTyping(false);
  if (data.error) { setStatus(data.error, true); return; }
  // For typed/browser-recognized input the user message is already shown.
  if (source === "server-voice") {
    addMessage(conv, { role: "user", text: data.recognized, ts: Date.now(), via: "voice" });
  }
  const botMsg = {
    role: "bot", text: data.response, ts: Date.now(),
    intent: data.intent, confidence: data.confidence, source: data.source, url: data.url || null,
  };
  if (conv.id === activeId) {
    showRecognized(data.recognized);
    showIntent(botMsg);
  }
  if (data.stream_id) { streamReply(conv, botMsg, data.stream_id); return; }
  addMessage(conv, botMsg);
  setStatus(DEFAULT_STATUS);
  if (ttsOn) speak(data.response);
}

// Open-ended answers from the generative model arrive as a text stream.
async function streamReply(conv, botMsg, streamId) {
  let live = null;
  const liveText = () => live && live.querySelector(".bubble > span");
  if (conv.id === activeId) {
    if (messagesEl.querySelector(".empty-state")) messagesEl.innerHTML = "";
    live = bubble({ ...botMsg, text: "" });
    liveText().innerHTML = `<span class="dots"><span>•</span><span>•</span><span>•</span></span>`;
    messagesEl.appendChild(live);
    scrollToBottom(true);
  }
  setStatus("Willow is writing...");
  let text = "";
  try {
    const res = await fetch(`/stream/${streamId}`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      text += decoder.decode(value, { stream: true });
      if (live && conv.id === activeId) {
        liveText().textContent = text.trimStart();
        scrollToBottom();
      }
    }
    text += decoder.decode();
  } catch {
    if (!text) text = "Sorry, I lost my train of thought. Please try again.";
  }
  if (live) live.remove();
  botMsg.text = text.trim() || "Sorry, I couldn't come up with an answer. Please try rephrasing.";
  botMsg.ts = Date.now();
  addMessage(conv, botMsg);
  setStatus(DEFAULT_STATUS);
  if (ttsOn) speak(botMsg.text);
}

async function sendText(text, via = "text") {
  text = text.trim();
  if (!text) return;
  const conv = activeConv();
  addMessage(conv, { role: "user", text, ts: Date.now(), via });
  showRecognized(text);
  showTyping(true);
  setStatus("Thinking...");
  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: conv.id, tz }),
    });
    handleReply(conv, await res.json(), via);
  } catch (e) {
    showTyping(false);
    setStatus("Couldn't reach Willow. Check your connection.", true);
  }
}

$("text-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const value = textInput.value;
  textInput.value = "";
  sendText(value);
});

$("new-chat").onclick = () => {
  if (draft && draft.id === activeId) { textInput.focus(); return; }
  startNewConversation();
  renderAll();
  textInput.focus();
};

// ---------- Read aloud toggle ----------
let ttsOn = readStore(TTS_KEY, true) !== false;
function syncTts() {
  ttsBtn.setAttribute("aria-pressed", String(ttsOn));
  ttsBtn.title = ttsOn ? "Reading replies aloud (click to mute)" : "Replies muted (click to read aloud)";
}
ttsBtn.onclick = () => {
  ttsOn = !ttsOn;
  if (!ttsOn) stopSpeaking();
  writeStore(TTS_KEY, ttsOn);
  syncTts();
};
if (!("speechSynthesis" in window)) ttsBtn.hidden = true;

// ---------- Willow's voice (female) + pause/resume ----------
const pauseBtn = $("pause");
const FEMALE_VOICES = [
  "Microsoft Aria Online", "Microsoft Jenny Online", "Microsoft Ava Online", "Microsoft Emma Online",
  "Google UK English Female", "Google US English", "Samantha", "Microsoft Zira", "Karen", "Moira",
  "Tessa", "Victoria", "Serena", "Fiona", "Allison", "Ava", "Susan", "Hazel", "Female",
];
let willowVoice = null;

function pickVoice() {
  if (!("speechSynthesis" in window)) return;
  const voices = speechSynthesis.getVoices().filter((v) => /^en/i.test(v.lang));
  for (const name of FEMALE_VOICES) {
    const v = voices.find((x) => x.name.includes(name));
    if (v) { willowVoice = v; return; }
  }
  willowVoice = null;
}
if ("speechSynthesis" in window) {
  pickVoice();
  speechSynthesis.onvoiceschanged = pickVoice;
}

function setPauseButton(state) { // "hidden" | "playing" | "paused"
  pauseBtn.hidden = state === "hidden";
  const paused = state === "paused";
  pauseBtn.innerHTML = icon(paused ? "play" : "pause") + `<span>${paused ? "Resume" : "Pause"}</span>`;
  pauseBtn.setAttribute("aria-label", paused ? "Resume reply" : "Pause reply");
  pauseBtn.title = paused ? "Resume reply" : "Pause reply";
}

function speak(text) {
  if (!("speechSynthesis" in window)) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  if (!willowVoice) pickVoice();
  if (willowVoice) { u.voice = willowVoice; u.lang = willowVoice.lang; }
  // Fallback when no named female voice exists: raise pitch for a lighter voice
  u.pitch = willowVoice ? 1.05 : 1.35;
  u.rate = 1;
  u.onstart = () => setPauseButton("playing");
  u.onend = u.onerror = () => setPauseButton("hidden");
  speechSynthesis.speak(u);
}

function stopSpeaking() {
  if ("speechSynthesis" in window) speechSynthesis.cancel();
  setPauseButton("hidden");
}

pauseBtn.onclick = () => {
  if (speechSynthesis.paused) { speechSynthesis.resume(); setPauseButton("playing"); }
  else { speechSynthesis.pause(); setPauseButton("paused"); }
};

// ---------- Waveform ----------
const waveEl = document.querySelector(".wave");
[10, 16, 12, 22, 14, 28, 18, 24, 30, 20, 14, 26, 32, 22, 16, 28, 20, 12, 24, 30, 18, 14, 22, 16, 12, 18, 10].forEach((h, i) => {
  const bar = document.createElement("i");
  bar.style.height = h + "px";
  bar.style.animationDelay = `${(i % 7) * -0.14}s`;
  waveEl.appendChild(bar);
});

// ---------- Speech recognition ----------
// Primary: browser Web Speech API (Chrome, Edge, Safari).
// Fallback: record WAV in the browser and transcribe on the server (/transcribe).
const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
let listening = false;

function setRecording(on) {
  listening = on;
  micBtn.classList.toggle("recording", on);
  micBtn.innerHTML = icon(on ? "stop" : "mic");
  micBtn.setAttribute("aria-label", on ? "Stop listening" : "Start speaking");
}

if (SpeechRec) {
  const rec = new SpeechRec();
  rec.lang = "en-US";
  rec.interimResults = true;
  rec.maxAlternatives = 1;
  let finalText = "";
  let errored = false;

  rec.onstart = () => {
    finalText = ""; errored = false;
    setRecording(true);
    setStatus("Listening... click to stop");
    showRecognized("Listening...", true);
  };
  rec.onresult = (e) => {
    let interim = "";
    for (let i = e.resultIndex; i < e.results.length; i++) {
      if (e.results[i].isFinal) finalText += e.results[i][0].transcript;
      else interim += e.results[i][0].transcript;
    }
    showRecognized((finalText + interim).trim() || "Listening...", true);
  };
  rec.onerror = (e) => {
    errored = true;
    setRecording(false);
    const messages = {
      "not-allowed": "Microphone access was blocked.",
      "service-not-allowed": "Microphone access was blocked.",
      "no-speech": "I didn't hear anything. Try again.",
      "audio-capture": "No microphone was found.",
      "network": "Speech service unavailable. Try typing instead.",
    };
    setStatus(messages[e.error] || `Speech error: ${e.error}`, true);
    renderInsights();
  };
  rec.onend = () => {
    setRecording(false);
    if (finalText.trim()) sendText(finalText, "voice");
    else if (!errored) { setStatus("Didn't catch anything. Try again.", true); renderInsights(); }
  };
  micBtn.onclick = () => {
    if (listening) { rec.stop(); return; }
    stopSpeaking();
    try { rec.start(); } catch { /* already starting */ }
  };
} else if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
  micBtn.onclick = () => { if (listening) stopWavRecording(); else { stopSpeaking(); startWavRecording(); } };
} else {
  micBtn.disabled = true;
  micBtn.title = "Voice input isn't supported in this browser";
  setStatus("Voice isn't supported here. Please type.", true);
}

// ---------- WAV recorder fallback ----------
let audioCtx, mediaStream, processor, chunks = [];

async function startWavRecording() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  } catch {
    setStatus("Microphone access was blocked.", true); return;
  }
  audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const src = audioCtx.createMediaStreamSource(mediaStream);
  processor = audioCtx.createScriptProcessor(4096, 1, 1);
  chunks = [];
  processor.onaudioprocess = (e) => chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  src.connect(processor);
  processor.connect(audioCtx.destination);
  setRecording(true);
  setStatus("Recording... click to stop");
  showRecognized("Recording...", true);
}

async function stopWavRecording() {
  setRecording(false);
  processor.disconnect();
  mediaStream.getTracks().forEach((t) => t.stop());
  const wav = encodeWav(chunks, audioCtx.sampleRate);
  audioCtx.close();
  const conv = activeConv();
  setStatus("Transcribing...");
  showRecognized("Transcribing...", true);
  showTyping(true);
  const form = new FormData();
  form.append("audio", wav, "speech.wav");
  form.append("session_id", conv.id);
  form.append("tz", tz);
  try {
    const res = await fetch("/transcribe", { method: "POST", body: form });
    const data = await res.json();
    if (data.error) renderInsights();
    handleReply(conv, data, "server-voice");
  } catch {
    showTyping(false);
    renderInsights();
    setStatus("Couldn't reach Willow. Check your connection.", true);
  }
}

function encodeWav(buffers, sampleRate) {
  const length = buffers.reduce((n, b) => n + b.length, 0);
  const view = new DataView(new ArrayBuffer(44 + length * 2));
  const writeStr = (o, s) => [...s].forEach((c, i) => view.setUint8(o + i, c.charCodeAt(0)));
  writeStr(0, "RIFF"); view.setUint32(4, 36 + length * 2, true); writeStr(8, "WAVE");
  writeStr(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  writeStr(36, "data"); view.setUint32(40, length * 2, true);
  let offset = 44;
  for (const b of buffers) for (let i = 0; i < b.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, b[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([view], { type: "audio/wav" });
}

// ---------- Boot ----------
if (!conversations.some((c) => c.id === activeId)) {
  if (conversations.length) activeId = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt)[0].id;
  else startNewConversation();
}
syncTts();
renderAll();
if (!storageOk) setStatus("History can't be saved in this browser.", true);

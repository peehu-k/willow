"""Rule-based skills that complement the neural intent classifier."""
import ast
import functools
import json
import operator
import re
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, available_timezones

# ---------- Maths ----------
_WORD_OPS = [
    (r"\bmultiplied by\b|\btimes\b|\binto\b|×|\bx\b", "*"),
    (r"\bdivided by\b|\bover\b|÷", "/"),
    (r"\bplus\b|\badded to\b", "+"),
    (r"\bminus\b|\bsubtract(?:ed)?\b", "-"),
    (r"\bto the power of\b|\braised to\b", "**"),
    (r"\bsquared\b", "**2"),
    (r"\bcubed\b", "**3"),
    (r"\bmod(?:ulo)?\b", "%"),
]
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


def try_math(text):
    """Return an answer string if the text is an arithmetic question, else None."""
    t = text.lower().replace(",", "")
    m = re.search(r"square root of (\d+(?:\.\d+)?)", t)
    if m:
        v = float(m.group(1)) ** 0.5
        return f"The square root of {m.group(1)} is {_fmt(v)}."
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:percent|%) of (\d+(?:\.\d+)?)", t)
    if m:
        return f"{m.group(1)} percent of {m.group(2)} is {_fmt(float(m.group(1)) * float(m.group(2)) / 100)}."
    for pattern, sym in _WORD_OPS:
        t = re.sub(pattern, f" {sym} ", t)
    expr = re.search(r"[\d.(][\d\s.+\-*/%()]*[\d)]", t)
    if not expr or not re.search(r"\d\s*(\*\*|[+\-*/%])\s*[\d(]", expr.group()):
        return None
    try:
        result = _safe_eval(ast.parse(expr.group(), mode="eval").body)
    except ZeroDivisionError:
        return "You can't divide by zero!"
    except Exception:
        return None
    pretty = re.sub(r"\s+", " ", expr.group()).replace("**", "^").strip()
    return f"{pretty} = {_fmt(result)}"


def _fmt(v):
    return str(int(v)) if float(v).is_integer() else f"{v:.4g}"


# ---------- Remember the user's name ----------
def extract_name(text):
    m = re.search(r"\b(?:my name is|call me|my name's|i am called)\s+([a-z]+)", text.lower())
    return m.group(1).capitalize() if m else None


# ---------- Wikipedia knowledge lookup ----------
_FILLER = r"^(?:(?:hey|hi|hello|ok|okay|so|and|but|um|uh|please|well|willow|what about|can you|could you|would you|do you know)\s+)*"
# Definitional / factual look-ups -> Wikipedia summary
_QUESTION = re.compile(
    _FILLER +
    r"(?:tell me (?:something )?about|who (?:is|was|are|were)|what (?:is|was|are|were)( an?| the)?|"
    r"what's|whats|who's|define|search for|search|look up|information (?:on|about)|"
    r"where (?:is|was)|when (?:is|was|did) )\s*(.+)$")
# "can you tell me what an alarm is" -> "alarm"
_EMBEDDED = re.compile(
    _FILLER + r"(?:tell me|explain|do you know|i want to know|i wonder) "
    r"(?:what|who) (?:an? |the )?(.+?) (?:is|are|was|were|means)$")
# Requests that need explanation, reasoning or creation -> generative model
_OPEN_ENDED = re.compile(
    r"\b(why|how (?:do|does|did|can|could|should|would|is|are|to|much|many|long|far)|explain|describe|"
    r"make|create|write|plan|planning|itinerary|suggest|recommend|list|compare|difference|differences|"
    r"recipe|steps|ideas|help me|should i|what if|invent(?:ed|or)?|discover(?:ed)?|summari[sz]e|"
    r"translate|advice|teach me|give me (?:a|an|some|ideas|examples|tips)|best way|pros and cons)\b", re.I)
_HEADERS = {"User-Agent": "WillowVoiceBot-StudentProject/1.0 (educational chatbot)"}


def _get_json(url):
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.load(r)


def _norm(text):
    return text.lower().strip(" ?.!")


def question_topic(text):
    """Topic of a definitional question ('what is X', 'tell me what X is'), else None."""
    t = _norm(text)
    m = _EMBEDDED.match(t)
    if m:
        return m.group(1)
    m = _QUESTION.match(t)
    return m.group(2) if m else None


def looks_like_question(text):
    """Definitional question about one topic. Relational ones ('the capital of France')
    need a direct answer, not an article summary, so they are left to the generative model."""
    topic = question_topic(text)
    return topic is not None and not re.match(r"^(?:the )?[a-z]+ (?:of|in|for|between) ", topic)


def is_open_ended(text):
    return bool(_OPEN_ENDED.search(text))


@functools.lru_cache(maxsize=512)
def _search(query, limit=1):
    data = _get_json("https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "query", "list": "search", "srsearch": query, "srlimit": limit, "format": "json"}))
    return tuple(h["title"] for h in data["query"]["search"])


@functools.lru_cache(maxsize=512)
def _summary(title):
    s = _get_json("https://en.wikipedia.org/api/rest_v1/page/summary/"
                  + urllib.parse.quote(title.replace(" ", "_")))
    if s.get("type") == "disambiguation" or not s.get("extract"):
        return None
    return s


def wikipedia_answer(text):
    """Search Wikipedia for the question topic and return a 2-sentence summary, or None."""
    query = question_topic(text) or text
    query = re.sub(r"\b(please|for me|exactly|actually|right now|in short|in simple words)\b", "", query).strip()
    if not query:
        return None
    try:
        summary = _direct_article(query)
        if summary == "ambiguous":
            return None  # e.g. "alarm": let the generative model give the everyday meaning
        if summary is None:
            titles = _search(query)
            summary = _summary(titles[0]) if titles else None
        if not summary:
            return None
        sentences = re.split(r"(?<=[.!?])\s+", summary["extract"])
        return {"title": summary.get("title", query), "answer": " ".join(sentences[:2]),
                "url": summary.get("content_urls", {}).get("desktop", {}).get("page")}
    except Exception:
        return None


@functools.lru_cache(maxsize=512)
def _direct_article(query):
    """Article whose title is exactly the topic; 'ambiguous' for disambiguation pages."""
    candidates = [query[:1].upper() + query[1:]]
    if len(query) <= 5:
        candidates.append(query.upper())  # acronyms: llm -> LLM
    for title in candidates:
        try:
            s = _get_json("https://en.wikipedia.org/api/rest_v1/page/summary/"
                          + urllib.parse.quote(title.replace(" ", "_")))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            raise
        if s.get("type") == "disambiguation":
            return "ambiguous"
        if s.get("extract"):
            return s
    return None


_STOP = set("a an the is are was were be to of in on at for and or but me my i you your can could would "
            "please tell what why how who when where which do does did there their it this that make "
            "create write give some about with like just ok okay so hey hi".split())


def wikipedia_notes(text, limit=1, max_chars=900):
    """Short reference notes from Wikipedia to ground the generative model (RAG)."""
    keywords = [w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _STOP]
    if not keywords:
        return None
    try:
        notes = []
        for title in _search(" ".join(keywords[:8]), limit):
            s = _summary(title)
            if s:
                notes.append(f"{title}: {s['extract'][:max_chars]}")
        return "\n".join(notes) or None
    except Exception:
        return None


# ---------- World clock ----------
_PLACE_TZ = {
    "india": "Asia/Kolkata", "delhi": "Asia/Kolkata", "new delhi": "Asia/Kolkata", "mumbai": "Asia/Kolkata",
    "bangalore": "Asia/Kolkata", "bengaluru": "Asia/Kolkata", "chennai": "Asia/Kolkata", "hyderabad": "Asia/Kolkata",
    "pune": "Asia/Kolkata", "dehradun": "Asia/Kolkata", "usa": "America/New_York", "america": "America/New_York",
    "new york": "America/New_York", "nyc": "America/New_York", "washington": "America/New_York",
    "california": "America/Los_Angeles", "san francisco": "America/Los_Angeles", "la": "America/Los_Angeles",
    "texas": "America/Chicago", "uk": "Europe/London", "england": "Europe/London", "britain": "Europe/London",
    "japan": "Asia/Tokyo", "china": "Asia/Shanghai", "beijing": "Asia/Shanghai", "australia": "Australia/Sydney",
    "canada": "America/Toronto", "germany": "Europe/Berlin", "france": "Europe/Paris", "italy": "Europe/Rome",
    "spain": "Europe/Madrid", "russia": "Europe/Moscow", "brazil": "America/Sao_Paulo", "uae": "Asia/Dubai",
    "pakistan": "Asia/Karachi", "nepal": "Asia/Kathmandu", "bangladesh": "Asia/Dhaka", "korea": "Asia/Seoul",
    "south korea": "Asia/Seoul", "singapore": "Asia/Singapore", "dubai": "Asia/Dubai", "mexico": "America/Mexico_City",
    "egypt": "Africa/Cairo", "south africa": "Africa/Johannesburg", "kenya": "Africa/Nairobi",
    "thailand": "Asia/Bangkok", "indonesia": "Asia/Jakarta", "new zealand": "Pacific/Auckland",
    "sri lanka": "Asia/Colombo", "saudi arabia": "Asia/Riyadh", "turkey": "Europe/Istanbul",
}
_CITY_TZ = {tz.split("/")[-1].replace("_", " ").lower(): tz for tz in available_timezones() if "/" in tz}


def world_time(text):
    """Answer 'what time is it in <place>' questions, else None."""
    m = re.search(r"\btime\b.*?\b(?:in|at)\s+([a-z .'-]+?)\s*(?:right now|now|currently|today)?[?.!]*$", text.lower())
    if not m:
        return None
    place = re.sub(r"^the\s+", "", m.group(1).strip(" .'"))
    tz = _PLACE_TZ.get(place) or _CITY_TZ.get(place)
    if not tz:
        return f"Sorry, I don't know the timezone for {place.title()}. Try a major city or country."
    now = datetime.now(timezone.utc).astimezone(ZoneInfo(tz))
    return f"It's {now.strftime('%I:%M %p').lstrip('0')} on {now.strftime('%A')} in {place.title()}."


# ---------- Opinion questions ----------
_OPINION = re.compile(
    r"(?:what do you think (?:about|of)|how do you feel about|do you like|do you love|"
    r"what(?:'s| is) your (?:opinion|view|take) (?:on|of|about)|your (?:opinion|thoughts) on)\s+(.+)")


def opinion_topic(text):
    m = _OPINION.search(text.lower().strip(" ?.!"))
    return m.group(1).strip() if m else None

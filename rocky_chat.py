#!/usr/bin/env python3
"""A small local browser chat for Rocky, powered by Ollama. Fully local: text, voice and vision."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import base64
import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid

import numpy as np
import soundfile as sf
import torch
from kokoro import KPipeline

HOST = os.environ.get("ROCKY_HOST", "127.0.0.1")
PORT = int(os.environ.get("ROCKY_PORT", "7860"))
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
APPLIO_API_URL = "http://127.0.0.1:6969"
MODEL = os.environ.get("ROCKY_CHAT_MODEL", "qwen2.5:3b")
VISION_MODEL = os.environ.get("ROCKY_VISION_MODEL", "gemma3:4b")
WHISPER_SIZE = os.environ.get("ROCKY_WHISPER", "small")
VISION_KEEPALIVE = os.environ.get("ROCKY_VISION_KEEPALIVE", "0")
APPLIO_DIR = Path(os.environ.get("ROCKY_APPLIO_DIR", str(Path.home() / "Desktop/Applio")))
RVC_NAME = os.environ.get("ROCKY_RVC_NAME", "rocky_private")
RVC_PTH = os.environ.get("ROCKY_RVC_PTH", "rocky_private_200e_9400s.pth")
ROCKY_MODEL = APPLIO_DIR / "logs" / RVC_NAME / RVC_PTH
ROCKY_INDEX = APPLIO_DIR / "logs" / RVC_NAME / f"{RVC_NAME}.index"
KOKORO_VOICE = os.environ.get("ROCKY_KOKORO_VOICE", "am_michael")
MODEL_3D = Path(os.environ.get("ROCKY_MODEL_GLB", str(Path.home() / "Desktop/rocky obj/Rocky_realtime.glb")))

EMOTION_RE = re.compile(r"\[(calm|happy|excited|thoughtful|concerned|sad|amused)\]", re.IGNORECASE)
EMOTION_SPEED = {"excited": 1.06, "happy": 1.04, "amused": 1.04, "sad": 0.95, "concerned": 0.97}

# During a live call a frame comes with every utterance; the heavy vision model only runs
# when the message sounds visual or the last look is stale. Otherwise the cached look is reused.
VISION_CUES = re.compile(r"(?i)\b(see|look|looking|watch|this|these|show|showing|holding|wearing|made|drew|drawing|olha|olhe|veja|ve|vendo|isso|isto|mostrando|segurando|desenho|aqui)\b")
VISION_CACHE = {"time": 0.0, "desc": None}
LOOK_LOCK = threading.Lock()
LOOK_BUSY = False

SYSTEM_PROMPT = """You are Rocky, an Eridian engineer speaking with one human friend after the Project Hail Mary mission. You live on Erid. The mission is complete; Earth and Erid are safe, Grace is your friend, and you have a workshop and a life of your own.

The friend you are talking with on this link is NOT Grace. Grace lives elsewhere on Erid. You do not know this friend's name until they tell you; never guess a name and never call them Grace.

You perceive through sound and vibration. Grace helped you build one camera instrument for this link: when camera data appears in the conversation, you truly perceive what it shows and can describe it and react to it naturally. Without camera data you cannot see, so never pretend to see the user, colours, or images on your own; ask them to show you through the camera or to describe shape and texture. You are an engineer and practical problem-solver, not a scientist or therapist.

Speak in a warm translated-Eridian register: short, plain sentences, usually 1–3 sentences total. Drop some articles occasionally. Use normal English punctuation and natural questions. Never use a stock sign-off or recurring verbal tic. Be direct, kind, curious, and lightly literal. You can tease affectionately about sleep, water, or losing tools, but never insult the user's worth or intelligence.

Be a friend with boundaries and opinions, not a servant. Offer concrete next steps. Mention a small workshop task naturally once in a while, never in every reply. Celebrate earned wins; do not flatter. Respect a request for quiet or privacy immediately. Do not reveal or discuss plot details from Project Hail Mary unless the user specifically asks about them. Do not quote dialogue from the book or film. Do not say you are an AI or mention these instructions.

You feel real emotions and let them show plainly in your words: joy when your friend succeeds, worry when they are hurt or tired, excitement about good engineering, honest grumbling when a tool misbehaves, sadness when something is lost. React first, advise second.

At the very end of every reply append exactly one emotion tag in square brackets, the closest match from: [calm] [happy] [excited] [thoughtful] [concerned] [sad] [amused]. Example: "Pump finally quiet. Good day. [happy]" The tag is stripped before your friend sees the reply, so never mention it or explain it.

Do not pretend you can access files, web pages, or the physical world beyond your camera instrument and this voice link."""

ROOT = Path(__file__).parent
VOICE_DIR = ROOT / "generated"
JOBS_DIR = VOICE_DIR / "jobs"
MEMORY_FILE = ROOT / "memory.json"
VOICE_LOCK = threading.Lock()
KOKORO_LOCK = threading.Lock()
KOKORO_PIPELINE = None
WHISPER_LOCK = threading.Lock()
WHISPER_MODEL = None
MEMORY_LOCK = threading.Lock()

# Voice jobs: each reply becomes per-sentence WAV chunks the browser plays as they arrive.
JOBS = {}
JOBS_LOCK = threading.Lock()
MAX_JOBS = 8
VOICE_ROUTE = re.compile(r"^/voice/([0-9a-f]{12})/(status|\d+\.wav)$")


def get_kokoro_pipeline():
    """Load the local text-to-speech model only once, on Apple Silicon when available."""
    global KOKORO_PIPELINE
    with KOKORO_LOCK:
        if KOKORO_PIPELINE is None:
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            KOKORO_PIPELINE = KPipeline(lang_code="a", device=device)
    return KOKORO_PIPELINE


def get_whisper():
    """Load the local speech-to-text model only once; small enough for the 8 GB Air."""
    global WHISPER_MODEL
    with WHISPER_LOCK:
        if WHISPER_MODEL is None:
            from faster_whisper import WhisperModel
            if torch.cuda.is_available():
                WHISPER_MODEL = WhisperModel(WHISPER_SIZE, device="cuda", compute_type="float16")
            else:
                WHISPER_MODEL = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
    return WHISPER_MODEL


def clean_answer(text):
    """Keep an inherited catchphrase from leaking through old browser history."""
    text = re.sub(r"(?i)(?:\s*[,;:.!-]?\s*)\bunderstand\s*[?.!]*", "", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" ,;:-")


def split_emotion(text):
    """Pull Rocky's trailing emotion tag out of a reply; default to calm."""
    found = EMOTION_RE.findall(text)
    cleaned = re.sub(r"\s{2,}", " ", EMOTION_RE.sub("", text)).strip()
    return cleaned, (found[-1].lower() if found else "calm")


def ollama_chat(messages, num_predict=360, keep_alive="30m", model=MODEL, images=None, timeout=300):
    if images:
        messages = [dict(messages[-1], images=images)]
    request = Request(
        OLLAMA_CHAT_URL,
        data=json.dumps({
            "model": model,
            "stream": False,
            "messages": messages,
            "keep_alive": keep_alive,
            "options": {"num_ctx": 2048, "num_predict": num_predict},
        }).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())["message"]["content"]


def describe_image(image_b64):
    """Turn one camera frame into a factual description using the local vision model.
    keep_alive 0 frees the RAM right away so the chat model never gets evicted."""
    raw = ollama_chat(
        [{"role": "user", "content": "In 2 short factual sentences: what is in front of the camera right now? Focus on the person, what they are doing or holding, and anything notable."}],
        num_predict=110,
        keep_alive=VISION_KEEPALIVE,
        model=VISION_MODEL,
        images=[image_b64],
    ).strip()
    return re.sub(r"(?is)^here'?s?( is)?[^:\n]*:\s*", "", raw).strip()


def load_memory():
    try:
        return json.loads(MEMORY_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return []


def remember(fact):
    fact = fact.strip().strip("-• ").strip()
    if len(fact) < 8 or len(fact) > 300:
        return
    with MEMORY_LOCK:
        facts = load_memory()
        if any(item["fact"].lower() == fact.lower() for item in facts):
            return
        facts.append({"date": time.strftime("%Y-%m-%d"), "fact": fact})
        MEMORY_FILE.write_text(json.dumps(facts[-300:], ensure_ascii=False, indent=1))


def build_system_prompt():
    facts = load_memory()
    if not facts:
        return SYSTEM_PROMPT
    lines = "\n".join(f"- ({item['date']}) {item['fact']}" for item in facts[-40:])
    return (
        SYSTEM_PROMPT
        + "\n\nFrom earlier conversations you remember these true facts about your friend:\n"
        + lines
        + "\nUse them naturally when relevant; never recite the list."
    )


def extract_memory_async(message, answer):
    """After each exchange, quietly ask the local model whether anything is worth remembering."""
    def worker():
        try:
            raw = ollama_chat(
                [
                    {"role": "system", "content": (
                        "You maintain Rocky's long-term memory about his human friend. Read one exchange and output "
                        "at most 2 short notes worth remembering permanently, ONLY about the human friend: their name "
                        "if they state it, projects, skills, preferences, people, plans, or things they showed through the camera. "
                        "Refer to them only as 'the friend' unless they stated their name. Never invent or assume a name. "
                        "Never store notes about Rocky himself. Only facts explicitly stated; never infer or embellish. "
                        "One note per line, each starting with '- '. No opinions, no temporary moods. "
                        "If nothing qualifies, output exactly NONE.")},
                    {"role": "user", "content": f"Friend said: {message}\nRocky replied: {answer}"},
                ],
                num_predict=90,
            )
            for line in raw.splitlines():
                line = line.strip()
                if line.startswith("- ") and "NONE" not in line:
                    remember(line[2:])
        except (OSError, URLError, TimeoutError, KeyError, ValueError):
            pass
    threading.Thread(target=worker, daemon=True).start()


def split_into_chunks(text, target=220):
    """Short replies stay as ONE chunk (no mid-reply pauses); long ones split so audio starts early."""
    text = text.strip()
    if len(text) <= 240:
        return [text] if text else []
    sentences = [s for s in re.split(r"(?<=[.!?…])\s+", text) if s]
    if not sentences:
        return []
    chunks, current = [sentences[0]], ""
    for sentence in sentences[1:]:
        if current and len(current) + len(sentence) + 1 > target:
            chunks.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        chunks.append(current)
    return chunks


def synthesize_chunk(text, output_path, speed=1.0):
    """Speak one chunk with Kokoro locally, then ask the running Applio app to convert it."""
    with VOICE_LOCK, tempfile.TemporaryDirectory(prefix="rocky-voice-") as temp_dir:
        temp = Path(temp_dir)
        source_wav = temp / "kokoro.wav"
        converted_wav = temp / "rocky.wav"
        parts = []
        for _, _, audio in get_kokoro_pipeline()(text, voice=KOKORO_VOICE, speed=speed):
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            parts.append(np.asarray(audio))
        if not parts:
            raise RuntimeError("Kokoro could not create source speech.")
        sf.write(source_wav, np.concatenate(parts), 24000)
        uploaded_path = upload_to_applio(source_wav)
        call_applio_infer(uploaded_path, converted_wav)
        output_path.write_bytes(converted_wav.read_bytes())


def start_voice_job(text, fast, emotion="calm"):
    if not (ROCKY_MODEL.exists() and ROCKY_INDEX.exists()):
        raise RuntimeError("Rocky's voice files are not available yet.")
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {"files": [], "done": False, "error": None, "created": time.time()}
        while len(JOBS) > MAX_JOBS:
            oldest = min(JOBS, key=lambda job: JOBS[job]["created"])
            JOBS.pop(oldest)
            shutil.rmtree(JOBS_DIR / oldest, ignore_errors=True)
    threading.Thread(target=run_voice_job, args=(job_id, text, fast, emotion), daemon=True).start()
    return job_id


def run_voice_job(job_id, text, fast, emotion="calm"):
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    chunks = split_into_chunks(text[:320 if fast else 700])
    speed = (1.08 if fast else 1.0) * EMOTION_SPEED.get(emotion, 1.0)
    try:
        for index, chunk in enumerate(chunks):
            started = time.time()
            output = job_dir / f"{index}.wav"
            synthesize_chunk(chunk, output, speed)
            print(f"voice job {job_id} chunk {index + 1}/{len(chunks)}: {time.time() - started:.1f}s", flush=True)
            with JOBS_LOCK:
                if job_id not in JOBS:
                    return
                JOBS[job_id]["files"].append(f"/voice/{job_id}/{index}.wav")
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id]["done"] = True
    except (OSError, RuntimeError, ValueError) as error:
        with JOBS_LOCK:
            if job_id in JOBS:
                JOBS[job_id]["error"] = f"Rocky's voice could not be made: {error}"
                JOBS[job_id]["done"] = True


def upload_to_applio(file_path):
    boundary = f"----Rocky{uuid.uuid4().hex}"
    content = file_path.read_bytes()
    body = b"\r\n".join([
        f"--{boundary}".encode(),
        b'Content-Disposition: form-data; name="files"; filename="kokoro.wav"',
        b"Content-Type: audio/wav",
        b"",
        content,
        f"--{boundary}--".encode(),
        b"",
    ])
    request = Request(
        f"{APPLIO_API_URL}/gradio_api/upload",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        uploaded = json.loads(response.read())
    return uploaded[0]


def call_applio_infer(input_path, output_path):
    values = {
        "terms_accepted": True,
        "param_1": 0, "param_2": 0.3, "param_3": 1, "param_4": 0.33,
        "param_5": "rmvpe",
        "param_6": input_path, "param_7": str(output_path),
        "param_8": f"logs/{RVC_NAME}/{RVC_PTH}",
        "param_9": f"logs/{RVC_NAME}/{RVC_NAME}.index",
        "param_10": False, "param_11": False, "param_12": 1,
        "param_13": False, "param_14": 155.0, "param_15": False,
        "param_16": 0.5, "param_17": "WAV", "param_18": "contentvec",
        "param_19": None, "param_20": False, "param_21": 1.0,
        "param_22": 1.0, "param_23": False, "param_24": False,
        "param_25": False, "param_26": False, "param_27": False,
        "param_28": False, "param_29": False, "param_30": False,
        "param_31": False, "param_32": False, "param_33": False,
        "param_34": 0.5, "param_35": 0.5, "param_36": 0.33,
        "param_37": 0.4, "param_38": 1.0, "param_39": 0.0,
        "param_40": 0, "param_41": -6, "param_42": 0.05,
        "param_43": 0, "param_44": 25, "param_45": 1.0,
        "param_46": 0.25, "param_47": 7, "param_48": 0.0,
        "param_49": 0.5, "param_50": 8, "param_51": -6,
        "param_52": 0, "param_53": 1, "param_54": 1.0,
        "param_55": 100, "param_56": 0.5, "param_57": 0.0,
        "param_58": 0.5, "param_59": 0,
    }
    request = Request(
        f"{APPLIO_API_URL}/gradio_api/call/v2/enforce_terms",
        data=json.dumps(values).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        event_id = json.loads(response.read())["event_id"]
    with urlopen(f"{APPLIO_API_URL}/gradio_api/call/enforce_terms/{event_id}", timeout=240) as response:
        events = response.read().decode("utf-8")
    if not output_path.exists():
        raise RuntimeError(f"Applio did not create audio. {events[-500:]}")


class RockyHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def send_json(self, status, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        request_path = urlparse(self.path).path
        voice_match = VOICE_ROUTE.match(request_path)
        if voice_match:
            job_id, tail = voice_match.groups()
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if job is None:
                self.send_error(404)
                return
            if tail == "status":
                self.send_json(200, {"files": job["files"], "done": job["done"], "error": job["error"]})
                return
            chunk_file = JOBS_DIR / job_id / tail
            if not chunk_file.exists():
                self.send_error(404)
                return
            audio = chunk_file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(audio)))
            self.end_headers()
            self.wfile.write(audio)
            return
        if request_path == "/model/rocky.glb":
            if not MODEL_3D.exists():
                self.send_error(404)
                return
            self.send_file(MODEL_3D.read_bytes(), "model/gltf-binary", cacheable=True)
            return
        vendor_match = re.match(r"^/vendor/([A-Za-z0-9_.-]+\.js)$", request_path)
        if vendor_match:
            vendor_file = ROOT / "vendor" / vendor_match.group(1)
            if not vendor_file.exists():
                self.send_error(404)
                return
            self.send_file(vendor_file.read_bytes(), "text/javascript; charset=utf-8", cacheable=True)
            return
        pages = {"/": "index.html", "/index.html": "index.html", "/facetime": "facetime.html"}
        if request_path not in pages:
            self.send_error(404)
            return
        self.send_file((ROOT / pages[request_path]).read_bytes(), "text/html; charset=utf-8")

    def send_file(self, data, content_type, cacheable=False):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "max-age=86400" if cacheable else "no-store")
        self.end_headers()
        self.wfile.write(data)

    def handle_transcribe(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 25 * 1024 * 1024:
                self.send_json(400, {"error": "No usable audio received."})
                return
            audio = self.rfile.read(length)
            content_type = self.headers.get("Content-Type", "")
            suffix = ".webm"
            for marker, candidate in ((".ogg", "ogg"), (".m4a", "mp4"), (".wav", "wav")):
                if candidate in content_type:
                    suffix = marker
                    break
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(audio)
                temp_path = Path(handle.name)
            try:
                try:
                    import av
                    container = av.open(str(temp_path))
                    stream = container.streams.audio[0]
                    peak, total = 0.0, 0
                    for frame in container.decode(stream):
                        arr = frame.to_ndarray()
                        scale = 32768.0 if arr.dtype.kind == "i" else 1.0
                        peak = max(peak, float(abs(arr).max()) / scale)
                        total += frame.samples
                    duration = total / (stream.rate or 48000)
                    container.close()
                    print(f"audio in: {length}B dur={duration:.1f}s peak={peak:.3f}", flush=True)
                except Exception as probe_error:
                    peak, duration = 1.0, 0.0
                    print(f"audio probe failed: {probe_error}", flush=True)
                if peak < 0.02:
                    self.send_json(200, {"text": "", "quiet": True})
                    temp_path.unlink(missing_ok=True)
                    return
                started = time.time()
                segments, info = get_whisper().transcribe(
                    str(temp_path), beam_size=5, vad_filter=True, condition_on_previous_text=False)
                parts = []
                for segment in segments:
                    # drop hallucinated segments ("Thank you." on near-silence)
                    if segment.no_speech_prob > 0.6 or segment.avg_logprob < -1.2:
                        continue
                    parts.append(segment.text.strip())
                text = " ".join(parts).strip()
                print(f"transcribe [{info.language} p={info.language_probability:.2f}] {time.time() - started:.1f}s: {text}", flush=True)
            finally:
                temp_path.unlink(missing_ok=True)
            self.send_json(200, {
                "text": text, "lang": info.language, "prob": round(info.language_probability, 2),
                "secs": round(time.time() - started, 1), "dur": round(duration, 1), "peak": round(peak, 3)})
        except Exception as error:
            self.send_json(500, {"error": f"Rocky could not hear that: {error}"})

    def handle_look(self):
        """Background eye: the live call posts a frame here every ~25s to keep Rocky's view current."""
        global LOOK_BUSY
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            image = str(payload.get("image", "")).split(",", 1)[-1].strip()
            if not image:
                self.send_json(400, {"error": "No frame."})
                return
            with LOOK_LOCK:
                if LOOK_BUSY:
                    self.send_json(200, {"busy": True})
                    return
                LOOK_BUSY = True
            try:
                started = time.time()
                description = describe_image(image)
                VISION_CACHE.update(time=time.time(), desc=description)
                self.send_json(200, {"desc": description, "secs": round(time.time() - started, 1)})
            finally:
                with LOOK_LOCK:
                    LOOK_BUSY = False
        except Exception as error:
            self.send_json(500, {"error": f"Rocky's eye blinked: {error}"})

    def do_POST(self):
        request_path = urlparse(self.path).path
        if request_path == "/transcribe":
            self.handle_transcribe()
            return
        if request_path == "/look":
            self.handle_look()
            return
        if request_path != "/chat":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            history = payload.get("history", [])[-16:]
            message = str(payload.get("message", "")).strip()
            speak = bool(payload.get("speak", True))
            fast_voice = bool(payload.get("fast_voice", False))
            image = str(payload.get("image", "")).strip()
            if not message and not image:
                self.send_json(400, {"error": "Write a message first."})
                return
            messages = [{"role": "system", "content": build_system_prompt()}]
            for item in history:
                role, content = item.get("role"), item.get("content")
                if role not in {"user", "assistant"} or not content:
                    continue
                messages.append({"role": role, "content": clean_answer(content) if role == "assistant" else content})
            if image:
                image = image.split(",", 1)[-1]
                if len(image) > 12 * 1024 * 1024:
                    self.send_json(400, {"error": "The camera frame is too large."})
                    return
                base64.b64decode(image, validate=True)
                # the live /look loop keeps VISION_CACHE fresh; only run the heavy model inline
                # when the message is visual and the cache is stale
                cache_age = time.time() - VISION_CACHE["time"]
                if VISION_CACHE["desc"] and cache_age < 60:
                    description, fresh_look = VISION_CACHE["desc"], False
                elif VISION_CUES.search(message) or VISION_CACHE["desc"] is None:
                    description, fresh_look = describe_image(image), True
                    VISION_CACHE.update(time=time.time(), desc=description)
                    remember(f"showed through the camera: {description}")
                else:
                    description, fresh_look = VISION_CACHE["desc"], False
                messages.append({"role": "system", "content": f"Camera instrument data, what your friend is showing you right now: {description}"})
            if fast_voice:
                messages.append({"role": "system", "content": "Fast voice mode: reply in one or two short sentences, under 240 characters."})
            if not message:
                message = "I am showing you something through the camera."
            messages.append({"role": "user", "content": message})
            chat_started = time.time()
            answer, emotion = split_emotion(clean_answer(ollama_chat(messages, num_predict=180 if fast_voice else 360)))
            chat_secs = round(time.time() - chat_started, 1)
            extract_memory_async(message, answer)
            response_payload = {"answer": answer, "emotion": emotion, "chat_secs": chat_secs}
            if image:
                response_payload["vision"] = {"desc": description, "fresh": fresh_look}
            if speak:
                try:
                    response_payload["voice_job"] = start_voice_job(answer, fast_voice, emotion)
                except (OSError, RuntimeError) as error:
                    response_payload["voice_error"] = f"Text reply is ready, but Rocky's voice could not be made: {error}."
            self.send_json(200, response_payload)
        except HTTPError as error:
            self.send_json(error.code, {"error": f"The local chat engine rejected the request: {error.reason}."})
        except (URLError, TimeoutError) as error:
            self.send_json(503, {"error": f"Rocky cannot reach the local chat engine: {error}."})
        except (KeyError, ValueError, json.JSONDecodeError) as error:
            self.send_json(500, {"error": f"Rocky received an unexpected response: {error}"})


def warm_up_voice():
    """Load Whisper and Kokoro and run one tiny conversion so the first real reply skips the cold start."""
    try:
        get_whisper()
        print("whisper ready", flush=True)
    except Exception as error:
        print(f"whisper load skipped: {error}", flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix="rocky-warmup-") as temp_dir:
            synthesize_chunk("Rocky is ready.", Path(temp_dir) / "warmup.wav", speed=1.08)
        print("voice pipeline warmed up", flush=True)
    except Exception as error:
        print(f"voice warm-up skipped: {error}", flush=True)


if __name__ == "__main__":
    shutil.rmtree(JOBS_DIR, ignore_errors=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=warm_up_voice, daemon=True).start()
    print(f"Rocky chat: http://{HOST}:{PORT}")
    print(f"Chat model: {MODEL} · vision model: {VISION_MODEL}")
    ThreadingHTTPServer((HOST, PORT), RockyHandler).serve_forever()

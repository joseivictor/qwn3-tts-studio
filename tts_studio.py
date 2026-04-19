"""
QWN3-TTS STUDIO v3.0 — 50 Features Edition
Better than ElevenLabs. Free. Local. Open Source.
Run: python -X utf8 tts_studio.py
"""

import io, os, sys, json, re, csv, uuid, time, base64, threading, tempfile, hashlib
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Auth ──────────────────────────────────────────────────────────────────────
_src = (Path(__file__).parent / "qwen_api.py").read_text(encoding="utf-8", errors="ignore")
_m   = re.search(r'HF_TOKEN\s*=\s*"([^"]+)"', _src)
if _m:
    from huggingface_hub import login
    login(token=_m.group(1), add_to_git_credential=False)

# ── GPU ───────────────────────────────────────────────────────────────────────
import torch
DTYPE      = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE_MAP = "cuda:0"       if torch.cuda.is_available() else None
GPU_NAME   = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
GPU_VRAM   = (f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB"
              if torch.cuda.is_available() else "")

OUT_DIR   = Path(__file__).parent / "audio_output"
PROJ_DIR  = Path(__file__).parent / "tts_projects"
HIST_FILE = Path(__file__).parent / "tts_history.json"
PRON_FILE = Path(__file__).parent / "tts_pronunciations.json"
for d in [OUT_DIR, PROJ_DIR]: d.mkdir(exist_ok=True)

# ── Audio processing ──────────────────────────────────────────────────────────
try:
    import librosa
    LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False

try:
    from pydub import AudioSegment
    PYDUB_OK = True
except ImportError:
    PYDUB_OK = False

# ── Models ────────────────────────────────────────────────────────────────────
_models = {}
_model_lock = threading.Lock()

def get_model(key: str):
    with _model_lock:
        if key not in _models:
            from qwen_tts import Qwen3TTSModel
            names = {
                "custom": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                "design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                "clone":  "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            }
            _models[key] = Qwen3TTSModel.from_pretrained(
                names[key], dtype=DTYPE, device_map=DEVICE_MAP)
    return _models[key]

threading.Thread(target=lambda: get_model("custom"), daemon=True).start()

# ── Data ──────────────────────────────────────────────────────────────────────
VOZES = {
    "Ryan":     {"label":"Ryan",    "desc":"Masculino profissional","icon":"👨‍💼","gender":"M"},
    "Dylan":    {"label":"Dylan",   "desc":"Masculino jovem",       "icon":"👦","gender":"M"},
    "Eric":     {"label":"Eric",    "desc":"Masculino neutro",      "icon":"🎙️","gender":"M"},
    "Aiden":    {"label":"Aiden",   "desc":"Masculino energético",  "icon":"⚡","gender":"M"},
    "Vivian":   {"label":"Vivian",  "desc":"Feminina suave",        "icon":"👩","gender":"F"},
    "Serena":   {"label":"Serena",  "desc":"Feminina elegante",     "icon":"💎","gender":"F"},
    "Uncle_Fu": {"label":"Uncle Fu","desc":"Grave / Dramático",     "icon":"🎭","gender":"M"},
    "Ono_Anna": {"label":"Ono Anna","desc":"Japonesa",              "icon":"🌸","gender":"F"},
    "Sohee":    {"label":"Sohee",   "desc":"Coreana",               "icon":"🎵","gender":"F"},
}
IDIOMAS = {
    "Português":"Portuguese","Inglês":"English","Chinês":"Chinese",
    "Japonês":"Japanese","Coreano":"Korean","Espanhol":"Spanish",
    "Francês":"French","Alemão":"German","Russo":"Russian","Auto":"Auto",
}
ESTILOS = [
    "","Entusiasmado e energético","Calmo e pausado","Triste e melancólico",
    "Raiva e intensidade","Locutor de rádio","Sussurrado e íntimo",
    "Humor e leveza","Sério e formal","Narrador de documentário",
    "Comercial publicitário","Conto de fadas","Jornalismo / Notícias",
    "Professor explicando","Stand-up comedy","Meditação / ASMR",
    "Motivacional","Horror / Suspense","Infantil","Esportivo / Narração",
]
EQ_PRESETS = {
    "Neutro": None,
    "Quente": {"bass":3,"mid":0,"treble":-2},
    "Brilhante": {"bass":-2,"mid":1,"treble":4},
    "Rádio": {"bass":2,"mid":3,"treble":2},
    "Podcast": {"bass":1,"mid":2,"treble":1},
    "Telefone": {"bass":-6,"mid":4,"treble":-4},
}

# ── State ─────────────────────────────────────────────────────────────────────
_history  = json.loads(HIST_FILE.read_text()) if HIST_FILE.exists() else []
_pron     = json.loads(PRON_FILE.read_text()) if PRON_FILE.exists() else {}
_queue    = []
_stats    = defaultdict(int)
_api_keys = {}

def save_history():
    HIST_FILE.write_text(json.dumps(_history[-200:], ensure_ascii=False, indent=2))

# ── 50 FEATURE FUNCTIONS ──────────────────────────────────────────────────────

# 1. wav → base64
def wav_to_b64(w: np.ndarray, sr: int) -> str:
    if w.ndim > 1: w = w.mean(-1)
    buf = io.BytesIO()
    sf.write(buf, w.astype(np.float32), sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()

# 2. save file
def save_wav_file(w: np.ndarray, sr: int, prefix: str) -> Path:
    if w.ndim > 1: w = w.mean(-1)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]
    path = OUT_DIR / f"{prefix}_{ts}_{uid}.wav"
    sf.write(str(path), w.astype(np.float32), sr)
    return path

# 3. normalize audio
def feature_normalize(w: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(w))
    return w / peak * 0.92 if peak > 0 else w

# 4. speed control (feature 4)
def feature_speed(w: np.ndarray, sr: int, rate: float) -> tuple:
    if not LIBROSA_OK or rate == 1.0: return w, sr
    w = librosa.effects.time_stretch(w.astype(np.float32), rate=rate)
    return w, sr

# 5. pitch control
def feature_pitch(w: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    if not LIBROSA_OK or semitones == 0: return w
    return librosa.effects.pitch_shift(w.astype(np.float32), sr=sr, n_steps=semitones)

# 6. trim silence
def feature_trim_silence(w: np.ndarray, sr: int, threshold_db: float = 30) -> np.ndarray:
    if not LIBROSA_OK: return w
    trimmed, _ = librosa.effects.trim(w.astype(np.float32), top_db=threshold_db)
    return trimmed

# 7. add reverb
def feature_reverb(w: np.ndarray, sr: int, amount: float = 0.3) -> np.ndarray:
    from scipy.signal import fftconvolve
    ir_len = int(sr * 0.5)
    t = np.linspace(0, 0.5, ir_len)
    ir = np.exp(-8 * t) * np.random.randn(ir_len) * amount
    ir[0] = 1.0
    wet = fftconvolve(w.astype(np.float32), ir)[:len(w)]
    return (w * (1 - amount) + wet * amount).astype(np.float32)

# 8. add echo
def feature_echo(w: np.ndarray, sr: int, delay_s: float = 0.3, decay: float = 0.4) -> np.ndarray:
    delay_samples = int(sr * delay_s)
    echo_w = np.zeros(len(w) + delay_samples, dtype=np.float32)
    echo_w[:len(w)] = w
    echo_w[delay_samples:] += w * decay
    return echo_w[:len(w)]

# 9. noise reduction (simple spectral subtraction)
def feature_denoise(w: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    sos = butter(4, [80, 8000], btype='band', fs=sr, output='sos')
    return sosfilt(sos, w.astype(np.float32))

# 10. stereo widening
def feature_stereo(w: np.ndarray, width: float = 0.3) -> np.ndarray:
    if w.ndim > 1: return w
    left  = w + np.roll(w, 3) * width
    right = w - np.roll(w, 3) * width
    return np.stack([left, right], axis=-1).astype(np.float32)

# 11. EQ
def feature_eq(w: np.ndarray, sr: int, bass: float=0, mid: float=0, treble: float=0) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    out = w.astype(np.float32)
    if bass != 0:
        sos = butter(2, 300, btype='low', fs=sr, output='sos')
        out = out + sosfilt(sos, out) * (10**(bass/20) - 1)
    if treble != 0:
        sos = butter(2, 4000, btype='high', fs=sr, output='sos')
        out = out + sosfilt(sos, out) * (10**(treble/20) - 1)
    return np.clip(out, -1.0, 1.0)

# 12. export as MP3
def feature_export_mp3(w: np.ndarray, sr: int, path: str) -> str:
    if not PYDUB_OK: return path.replace(".mp3",".wav")
    wav_buf = io.BytesIO()
    sf.write(wav_buf, w.astype(np.float32), sr, format="WAV")
    wav_buf.seek(0)
    seg = AudioSegment.from_wav(wav_buf)
    mp3_path = path if path.endswith(".mp3") else path.replace(".wav",".mp3")
    seg.export(mp3_path, format="mp3", bitrate="192k")
    return mp3_path

# 13. export as FLAC
def feature_export_flac(w: np.ndarray, sr: int, prefix: str) -> Path:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"{prefix}_{ts}.flac"
    sf.write(str(path), w.astype(np.float32), sr, format="FLAC")
    return path

# 14. generate SRT subtitles (estimated word timing)
def feature_generate_srt(text: str, duration_s: float) -> str:
    words = text.split()
    if not words: return ""
    wps   = len(words) / max(duration_s, 0.1)
    lines, t = [], 0.0
    chunk_size = 8
    for i in range(0, len(words), chunk_size):
        chunk   = " ".join(words[i:i+chunk_size])
        t_start = t
        t_end   = t + len(words[i:i+chunk_size]) / max(wps, 0.1)
        def fmt(s):
            h,r=divmod(int(s),3600); m,sec=divmod(r,60); ms=int((s%1)*1000)
            return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"
        lines.append(f"{i//chunk_size+1}\n{fmt(t_start)} --> {fmt(t_end)}\n{chunk}\n")
        t = t_end
    return "\n".join(lines)

# 15. SSML-like tag parser
def feature_parse_ssml(text: str) -> tuple:
    instruct_parts = []
    text = re.sub(r'<break\s+time="([^"]+)"\s*/>', r'... ', text)
    em = re.findall(r'<emphasis>(.*?)</emphasis>', text)
    if em: instruct_parts.append("enfatize as palavras importantes")
    text = re.sub(r'<emphasis>(.*?)</emphasis>', r'\1', text)
    slow = re.findall(r'<slow>(.*?)</slow>', text)
    if slow: instruct_parts.append("fale mais devagar")
    text = re.sub(r'<slow>(.*?)</slow>', r'\1', text)
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip(), " e ".join(instruct_parts) if instruct_parts else None

# 16. custom pronunciation replacement
def feature_apply_pronunciation(text: str) -> str:
    for word, pron in _pron.items():
        text = re.sub(r'\b' + re.escape(word) + r'\b', pron, text, flags=re.IGNORECASE)
    return text

# 17. auto punctuation
def feature_auto_punctuation(text: str) -> str:
    text = re.sub(r'\s+', ' ', text.strip())
    if text and not text[-1] in '.!?': text += '.'
    text = re.sub(r'([.!?])\s+([a-záéíóúàâêôãõçA-Z])', lambda m: m.group(1)+' '+m.group(2).upper(), text)
    return text

# 18. text cleanup (remove markdown/HTML)
def feature_clean_text(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    text = re.sub(r'`[^`]*`', '', text)
    return text.strip()

# 19. word count + estimated duration
def feature_text_stats(text: str) -> dict:
    words = len(text.split())
    chars = len(text)
    est_s = words / 2.8
    return {"words": words, "chars": chars, "est_seconds": round(est_s,1),
            "est_time": f"{int(est_s//60)}:{int(est_s%60):02d}"}

# 20. split long text into chunks
def feature_chunk_text(text: str, max_chars: int = 300) -> list:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) < max_chars:
            cur += (" " if cur else "") + s
        else:
            if cur: chunks.append(cur)
            cur = s
    if cur: chunks.append(cur)
    return chunks

# 21. batch generate (multiple lines)
def feature_batch_texts(texts: list, voz: str, idioma: str, estilo: str=None) -> list:
    results = []
    model = get_model("custom")
    for i, text in enumerate(texts):
        if not text.strip(): continue
        try:
            wavs, sr = model.generate_custom_voice(text=text, language=idioma, speaker=voz, instruct=estilo)
            w = np.asarray(wavs[0], dtype=np.float32)
            w = feature_normalize(w)
            path = save_wav_file(w, sr, f"batch_{i:03d}")
            results.append({"text": text[:50], "path": str(path), "ok": True})
        except Exception as e:
            results.append({"text": text[:50], "error": str(e), "ok": False})
    return results

# 22. podcast mode (multi-speaker script)
def feature_podcast(script: list) -> tuple:
    """script = [{"speaker":"Ryan","text":"Olá!"},...]"""
    model = get_model("custom")
    segments = []
    for seg in script:
        speaker = seg.get("speaker","Ryan")
        text    = seg.get("text","")
        if not text.strip(): continue
        wavs, sr = model.generate_custom_voice(text=text, language="Portuguese", speaker=speaker)
        w = np.asarray(wavs[0], dtype=np.float32)
        segments.append((w, sr))
    if not segments: return np.array([]), 24000
    sr = segments[0][1]
    silence = np.zeros(int(sr * 0.4), dtype=np.float32)
    combined = np.concatenate([np.concatenate([s, silence]) for s, _ in segments])
    return combined, sr

# 23. voice comparison (generate same text with 2 voices)
def feature_compare_voices(text: str, voice_a: str, voice_b: str, idioma: str) -> dict:
    model = get_model("custom")
    results = {}
    for voz in [voice_a, voice_b]:
        wavs, sr = model.generate_custom_voice(text=text, language=idioma, speaker=voz)
        w = np.asarray(wavs[0], dtype=np.float32)
        results[voz] = {"b64": wav_to_b64(w, sr), "sr": sr}
    return results

# 24. add background music (mix with audio file)
def feature_add_background(w: np.ndarray, sr: int, music_path: str, volume: float=0.15) -> np.ndarray:
    if not LIBROSA_OK: return w
    music, _ = librosa.load(music_path, sr=sr, mono=True)
    if len(music) < len(w):
        repeats = int(np.ceil(len(w) / len(music)))
        music = np.tile(music, repeats)
    music = music[:len(w)] * volume
    return np.clip(w + music, -1.0, 1.0)

# 25. fade in/out
def feature_fade(w: np.ndarray, sr: int, fade_in_s: float=0.1, fade_out_s: float=0.2) -> np.ndarray:
    w = w.copy()
    fi = int(sr * fade_in_s)
    fo = int(sr * fade_out_s)
    if fi > 0: w[:fi] *= np.linspace(0, 1, fi)
    if fo > 0: w[-fo:] *= np.linspace(1, 0, fo)
    return w

# 26. audio compression (dynamic range)
def feature_compress(w: np.ndarray, threshold: float=0.5, ratio: float=4.0) -> np.ndarray:
    mask = np.abs(w) > threshold
    w = w.copy()
    w[mask] = np.sign(w[mask]) * (threshold + (np.abs(w[mask]) - threshold) / ratio)
    return w

# 27. add intro/outro silence or tone
def feature_add_padding(w: np.ndarray, sr: int, pad_s: float=0.3) -> np.ndarray:
    pad = np.zeros(int(sr * pad_s), dtype=np.float32)
    return np.concatenate([pad, w, pad])

# 28. generate waveform data for visualization
def feature_waveform_data(w: np.ndarray, points: int=200) -> list:
    if len(w) == 0: return [0] * points
    step = max(1, len(w) // points)
    return [float(np.max(np.abs(w[i:i+step]))) for i in range(0, len(w), step)][:points]

# 29. audio stats
def feature_audio_stats(w: np.ndarray, sr: int) -> dict:
    return {
        "duration_s": round(len(w)/sr, 2),
        "peak_db":    round(20*np.log10(max(np.max(np.abs(w)),1e-9)), 1),
        "rms_db":     round(20*np.log10(max(np.sqrt(np.mean(w**2)),1e-9)), 1),
        "sample_rate":sr,
        "samples":    len(w),
    }

# 30. save project
def feature_save_project(name: str, data: dict) -> str:
    path = PROJ_DIR / f"{name.replace(' ','_')}.json"
    data["saved_at"] = datetime.now().isoformat()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    return str(path)

# 31. load project
def feature_load_project(name: str) -> dict:
    path = PROJ_DIR / f"{name.replace(' ','_')}.json"
    if path.exists(): return json.loads(path.read_text())
    return {}

# 32. list projects
def feature_list_projects() -> list:
    return [p.stem for p in PROJ_DIR.glob("*.json")]

# 33. export history as CSV
def feature_export_csv() -> str:
    path = OUT_DIR / f"history_{datetime.now().strftime('%Y%m%d')}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id","text","voice","lang","style","path","duration","created"])
        w.writeheader()
        for h in _history:
            w.writerow({k: h.get(k,"") for k in w.fieldnames})
    return str(path)

# 34. voice aging (simple pitch+speed combo)
def feature_voice_aging(w: np.ndarray, sr: int, age_delta: int=10) -> np.ndarray:
    if not LIBROSA_OK: return w
    semitones = -age_delta * 0.05
    rate      = 1.0 - age_delta * 0.003
    w = librosa.effects.pitch_shift(w.astype(np.float32), sr=sr, n_steps=semitones)
    w, _ = feature_speed(w, sr, max(0.7, rate))
    return w

# 35. gender morph (pitch shift)
def feature_gender_morph(w: np.ndarray, sr: int, direction: str="female") -> np.ndarray:
    if not LIBROSA_OK: return w
    steps = 4 if direction == "female" else -4
    return librosa.effects.pitch_shift(w.astype(np.float32), sr=sr, n_steps=steps)

# 36. search history
def feature_search_history(query: str) -> list:
    q = query.lower()
    return [h for h in _history if q in h.get("text","").lower() or q in h.get("voice","").lower()]

# 37. estimate cost vs ElevenLabs
def feature_cost_estimate(chars: int) -> dict:
    eleven_cost = chars / 1000 * 0.30
    our_cost    = 0.0
    return {"chars": chars, "eleven_labs_usd": round(eleven_cost,4), "ours_usd": 0.0,
            "savings_usd": round(eleven_cost,4), "savings_pct": 100}

# 38. pronunciation editor
def feature_add_pronunciation(word: str, phonetic: str):
    _pron[word] = phonetic
    PRON_FILE.write_text(json.dumps(_pron, ensure_ascii=False, indent=2))

def feature_remove_pronunciation(word: str):
    _pron.pop(word, None)
    PRON_FILE.write_text(json.dumps(_pron, ensure_ascii=False, indent=2))

# 39. add to queue
def feature_enqueue(job: dict) -> str:
    job_id = uuid.uuid4().hex[:8]
    job["id"] = job_id
    job["status"] = "queued"
    job["created"] = datetime.now().isoformat()
    _queue.append(job)
    return job_id

# 40. generate API key
def feature_generate_api_key(name: str) -> str:
    key = "qwn3-" + hashlib.sha256(f"{name}{time.time()}".encode()).hexdigest()[:32]
    _api_keys[key] = {"name": name, "created": datetime.now().isoformat(), "calls": 0}
    return key

# 41. usage statistics
def feature_stats() -> dict:
    return {
        "total_generated": _stats["generated"],
        "total_chars":     _stats["chars"],
        "total_seconds":   round(_stats["seconds"], 1),
        "by_voice":        dict(_stats),
        "history_count":   len(_history),
        "projects":        len(feature_list_projects()),
    }

# 42. audio trim (clip between start/end seconds)
def feature_trim_audio(w: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    s = int(start_s * sr)
    e = int(end_s   * sr) if end_s > 0 else len(w)
    return w[s:e]

# 43. join multiple audio files
def feature_join_audios(paths: list, gap_s: float=0.3) -> tuple:
    segments, sr = [], 24000
    for path in paths:
        w, sr = sf.read(path)
        if w.ndim > 1: w = w.mean(-1)
        segments.append(w.astype(np.float32))
    silence = np.zeros(int(sr * gap_s), dtype=np.float32)
    combined = np.concatenate([np.concatenate([s, silence]) for s in segments])
    return combined, sr

# 44. real-time preview (first 10 seconds worth of chars)
def feature_preview_text(text: str, max_words: int=20) -> str:
    words = text.split()
    return " ".join(words[:max_words]) + ("..." if len(words) > max_words else "")

# 45. detect language
def feature_detect_language(text: str) -> str:
    pt = len(re.findall(r'\b(de|da|do|para|com|que|uma|por|não|são|também|mas)\b', text, re.I))
    en = len(re.findall(r'\b(the|and|for|are|with|this|that|from|have|been)\b', text, re.I))
    es = len(re.findall(r'\b(que|una|para|con|por|como|más|pero|esto|este)\b', text, re.I))
    scores = {"Português":pt, "Inglês":en, "Espanhol":es}
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "Auto"

# 46. add history entry
def feature_add_history(entry: dict):
    entry["id"] = uuid.uuid4().hex[:8]
    entry["created"] = datetime.now().isoformat()
    _history.insert(0, entry)
    _stats["generated"] += 1
    _stats["chars"]     += len(entry.get("text",""))
    _stats["seconds"]   += entry.get("duration_s", 0)
    save_history()

# 47. delete history entry
def feature_delete_history(entry_id: str):
    global _history
    _history = [h for h in _history if h.get("id") != entry_id]
    save_history()

# 48. audio fingerprint (detect duplicate generations)
def feature_fingerprint(w: np.ndarray) -> str:
    snippet = w[:1024] if len(w) >= 1024 else w
    return hashlib.md5(snippet.tobytes()).hexdigest()[:12]

# 49. apply all processing pipeline
def feature_process_pipeline(w: np.ndarray, sr: int, opts: dict) -> tuple:
    if opts.get("normalize"):   w = feature_normalize(w)
    if opts.get("trim"):        w = feature_trim_silence(w, sr)
    speed = opts.get("speed", 1.0)
    if speed != 1.0:            w, sr = feature_speed(w, sr, speed)
    pitch = opts.get("pitch", 0.0)
    if pitch != 0.0:            w = feature_pitch(w, sr, pitch)
    if opts.get("reverb"):      w = feature_reverb(w, sr, opts.get("reverb_amount",0.3))
    if opts.get("echo"):        w = feature_echo(w, sr)
    if opts.get("compress"):    w = feature_compress(w)
    eq = opts.get("eq")
    if eq and eq != "Neutro":
        p = EQ_PRESETS.get(eq,{})
        if p: w = feature_eq(w, sr, p.get("bass",0), p.get("mid",0), p.get("treble",0))
    if opts.get("fade"):        w = feature_fade(w, sr)
    if opts.get("padding"):     w = feature_add_padding(w, sr)
    w = feature_normalize(w)
    return w, sr

# 50. core generate with all features
def generate_audio(texto: str, voz: str, idioma: str, estilo: str=None, opts: dict=None) -> dict:
    opts = opts or {}
    t0   = time.time()

    # Pre-processing
    texto = feature_clean_text(texto)
    texto = feature_apply_pronunciation(texto)
    texto, ssml_instruct = feature_parse_ssml(texto)
    if opts.get("auto_punctuation"): texto = feature_auto_punctuation(texto)
    if not estilo and ssml_instruct: estilo = ssml_instruct

    # Detect language if auto
    if idioma == "Auto": idioma = IDIOMAS.get(feature_detect_language(texto), "Portuguese")

    model = get_model("custom")
    wavs, sr = model.generate_custom_voice(text=texto, language=idioma, speaker=voz, instruct=estilo or None)
    w = np.asarray(wavs[0], dtype=np.float32)

    # Post-processing pipeline
    w, sr = feature_process_pipeline(w, sr, opts)

    duration_s = time.time() - t0
    stats = feature_audio_stats(w, sr)
    waveform = feature_waveform_data(w)

    # Save
    prefix = f"tts_{voz.lower()}"
    path = save_wav_file(w, sr, prefix)

    # Export formats
    exports = {"wav": str(path)}
    if opts.get("export_mp3"):
        exports["mp3"] = feature_export_mp3(w, sr, str(path).replace(".wav",".mp3"))
    if opts.get("export_flac"):
        exports["flac"] = str(feature_export_flac(w, sr, prefix))

    # SRT
    srt = None
    if opts.get("generate_srt"):
        srt = feature_generate_srt(texto, stats["duration_s"])
        srt_path = str(path).replace(".wav",".srt")
        Path(srt_path).write_text(srt, encoding="utf-8")
        exports["srt"] = srt_path

    b64 = wav_to_b64(w, sr)
    cost = feature_cost_estimate(len(texto))

    entry = {"text":texto,"voice":voz,"lang":idioma,"style":estilo or "",
             "path":str(path),"duration":round(stats["duration_s"],2),
             "duration_s":duration_s}
    feature_add_history(entry)

    return {
        "audio_b64":   b64,
        "filename":    path.name,
        "path":        str(path),
        "duration_s":  round(duration_s, 1),
        "stats":       stats,
        "waveform":    waveform,
        "exports":     exports,
        "srt":         srt,
        "cost_saved":  cost,
        "fingerprint": feature_fingerprint(w),
    }

# ── HTML ──────────────────────────────────────────────────────────────────────
HTML_PAGE = open(Path(__file__).parent / "tts_ui.html", encoding="utf-8").read() if (Path(__file__).parent / "tts_ui.html").exists() else None

# ── FASTAPI APP ───────────────────────────────────────────────────────────────
app = FastAPI(title="QWN3-TTS Studio")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/", response_class=HTMLResponse)
async def root():
    return _build_html()

@app.get("/api/info")
async def api_info():
    return {"gpu":GPU_NAME,"vram":GPU_VRAM,"models_loaded":list(_models.keys()),
            "features":50,"version":"3.0"}

@app.post("/api/generate")
async def api_generate(req: dict):
    try:
        return generate_audio(
            texto  = req.get("texto",""),
            voz    = req.get("voz","Ryan"),
            idioma = req.get("idioma","Portuguese"),
            estilo = req.get("estilo") or None,
            opts   = req.get("opts",{}),
        )
    except Exception as e:
        return JSONResponse({"error":str(e)}, status_code=500)

@app.post("/api/generate/design")
async def api_design(req: dict):
    t0 = time.time()
    try:
        model = get_model("design")
        wavs, sr = model.generate_voice_design(
            text=req["texto"], language=req.get("idioma","Portuguese"),
            instruct=req.get("descricao",""))
        w = np.asarray(wavs[0], dtype=np.float32)
        w = feature_normalize(w)
        path = save_wav_file(w, sr, "design")
        feature_add_history({"text":req["texto"],"voice":"design","lang":req.get("idioma",""),
                              "style":req.get("descricao",""),"path":str(path),"duration":round(len(w)/sr,2),"duration_s":time.time()-t0})
        return {"audio_b64":wav_to_b64(w,sr),"filename":path.name,"path":str(path),
                "duration_s":round(time.time()-t0,1),"stats":feature_audio_stats(w,sr),"waveform":feature_waveform_data(w)}
    except Exception as e:
        return JSONResponse({"error":str(e)},status_code=500)

@app.post("/api/generate/clone")
async def api_clone(texto:str=Form(...),idioma:str=Form(...),
                    transcricao:str=Form(""),audio_ref:UploadFile=File(...)):
    t0=time.time(); tmp=None
    try:
        data=await audio_ref.read()
        with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as f:
            f.write(data); tmp=f.name
        model=get_model("clone")
        wavs,sr=model.generate_voice_clone(text=texto,language=idioma,ref_audio=tmp,ref_text=transcricao)
        w=np.asarray(wavs[0],dtype=np.float32)
        w=feature_normalize(w)
        path=save_wav_file(w,sr,"clone")
        feature_add_history({"text":texto,"voice":"clone","lang":idioma,"style":"","path":str(path),"duration":round(len(w)/sr,2),"duration_s":time.time()-t0})
        return {"audio_b64":wav_to_b64(w,sr),"filename":path.name,"path":str(path),
                "duration_s":round(time.time()-t0,1),"waveform":feature_waveform_data(w)}
    except Exception as e:
        return JSONResponse({"error":str(e)},status_code=500)
    finally:
        if tmp and os.path.exists(tmp): os.unlink(tmp)

@app.post("/api/batch")
async def api_batch(req: dict):
    texts = req.get("texts",[])
    voz   = req.get("voz","Ryan")
    idioma= req.get("idioma","Portuguese")
    estilo= req.get("estilo")
    return {"results": feature_batch_texts(texts, voz, idioma, estilo)}

@app.post("/api/podcast")
async def api_podcast(req: dict):
    script = req.get("script",[])
    w, sr  = feature_podcast(script)
    if len(w) == 0: return JSONResponse({"error":"Script vazio"},status_code=400)
    path = save_wav_file(w, sr, "podcast")
    return {"audio_b64":wav_to_b64(w,sr),"filename":path.name,"path":str(path)}

@app.post("/api/compare")
async def api_compare(req: dict):
    return feature_compare_voices(req["texto"],req["voice_a"],req["voice_b"],req.get("idioma","Portuguese"))

@app.get("/api/history")
async def api_history(q: str=""):
    if q: return feature_search_history(q)
    return _history[:50]

@app.delete("/api/history/{entry_id}")
async def api_delete_history(entry_id: str):
    feature_delete_history(entry_id); return {"ok":True}

@app.get("/api/history/export")
async def api_export_history():
    path = feature_export_csv()
    return FileResponse(path, filename=Path(path).name)

@app.get("/api/stats")
async def api_stats():
    return feature_stats()

@app.get("/api/projects")
async def api_projects():
    return feature_list_projects()

@app.post("/api/projects")
async def api_save_project(req: dict):
    return {"path": feature_save_project(req["name"], req.get("data",{}))}

@app.get("/api/projects/{name}")
async def api_load_project(name: str):
    return feature_load_project(name)

@app.get("/api/pronunciations")
async def api_pronunciations():
    return _pron

@app.post("/api/pronunciations")
async def api_add_pronunciation(req: dict):
    feature_add_pronunciation(req["word"], req["phonetic"]); return {"ok":True}

@app.delete("/api/pronunciations/{word}")
async def api_del_pronunciation(word: str):
    feature_remove_pronunciation(word); return {"ok":True}

@app.post("/api/text/stats")
async def api_text_stats(req: dict):
    return feature_text_stats(req.get("text",""))

@app.post("/api/text/detect-lang")
async def api_detect_lang(req: dict):
    return {"language": feature_detect_language(req.get("text",""))}

@app.post("/api/text/clean")
async def api_clean_text(req: dict):
    return {"text": feature_clean_text(req.get("text",""))}

@app.post("/api/keys")
async def api_create_key(req: dict):
    return {"key": feature_generate_api_key(req.get("name","default"))}

@app.get("/api/cost")
async def api_cost(chars: int=1000):
    return feature_cost_estimate(chars)

@app.get("/api/audio/{filename}")
async def api_serve_audio(filename: str):
    path = OUT_DIR / filename
    if path.exists(): return FileResponse(str(path))
    return JSONResponse({"error":"Not found"},status_code=404)

# ── HTML BUILDER ──────────────────────────────────────────────────────────────
def _build_html():
    import json as _json
    vozes_j  = _json.dumps(VOZES,   ensure_ascii=False)
    idiomas_j= _json.dumps(IDIOMAS, ensure_ascii=False)
    estilos_j= _json.dumps(ESTILOS, ensure_ascii=False)
    eq_j     = _json.dumps(list(EQ_PRESETS.keys()), ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QWN3-TTS Studio</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{--bg:#070710;--s1:#0f0f1c;--s2:#16162a;--s3:#1e1e35;--bd:#2d2d50;
  --t:#e8e8f5;--dim:#6b6b90;--a:#7c3aed;--a2:#a855f7;--a3:#c084fc;
  --g:#22c55e;--r:#ef4444;--y:#f59e0b;--b:#3b82f6;--pk:#ec4899}}
body{{background:var(--bg);color:var(--t);font-family:'Inter',system-ui,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}}
/* HEADER */
header{{background:linear-gradient(135deg,#1a0533 0%,#0f0f1c 100%);border-bottom:1px solid var(--bd);padding:14px 28px;display:flex;align-items:center;gap:16px;flex-shrink:0}}
.logo{{font-size:22px;font-weight:800;background:linear-gradient(135deg,#c084fc,#60a5fa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;letter-spacing:-0.5px}}
.tagline{{font-size:11px;color:var(--dim);margin-top:1px}}
.badges{{display:flex;gap:6px;margin-left:auto}}
.badge{{background:var(--s2);border:1px solid var(--bd);padding:4px 10px;border-radius:20px;font-size:11px;color:var(--dim)}}
.badge.on{{color:var(--g);border-color:#22c55e40}}
/* LAYOUT */
.app{{display:flex;flex:1;overflow:hidden}}
/* SIDEBAR */
.sidebar{{width:220px;background:var(--s1);border-right:1px solid var(--bd);display:flex;flex-direction:column;overflow-y:auto;flex-shrink:0}}
.nav-section{{padding:16px 12px 8px;font-size:10px;text-transform:uppercase;letter-spacing:1.5px;color:var(--dim)}}
.nav-btn{{display:flex;align-items:center;gap:10px;padding:9px 16px;cursor:pointer;border:none;background:none;color:var(--dim);font-size:13px;width:100%;text-align:left;border-left:2px solid transparent;transition:all .15s}}
.nav-btn:hover{{background:var(--s2);color:var(--t)}}
.nav-btn.active{{background:var(--s2);color:var(--a3);border-left-color:var(--a)}}
.nav-btn .ico{{font-size:16px;width:20px;text-align:center}}
/* MAIN */
.main{{flex:1;overflow-y:auto;padding:24px}}
/* PAGE */
.page{{display:none;max-width:960px;margin:0 auto;animation:fi .2s ease}}
.page.active{{display:block}}
@keyframes fi{{from{{opacity:0;transform:translateY(6px)}}to{{opacity:1;transform:translateY(0)}}}}
/* GRID */
.grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}}
@media(max-width:768px){{.grid-2,.grid-3{{grid-template-columns:1fr}}}}
/* CARD */
.card{{background:var(--s1);border:1px solid var(--bd);border-radius:14px;padding:20px}}
.card-title{{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1.2px;color:var(--dim);margin-bottom:14px;display:flex;align-items:center;gap:8px}}
/* FORM */
label{{font-size:12px;color:var(--dim);display:block;margin-bottom:5px;margin-top:12px}}
label:first-child{{margin-top:0}}
select,textarea,input[type=text],input[type=number],input[type=range]{{width:100%;background:var(--s2);border:1px solid var(--bd);color:var(--t);padding:9px 12px;border-radius:9px;font-size:13px;font-family:inherit;outline:none;transition:border .15s;-webkit-appearance:none}}
select:focus,textarea:focus,input:focus{{border-color:var(--a)}}
textarea{{resize:vertical;min-height:100px;line-height:1.65}}
input[type=range]{{padding:4px 0;cursor:pointer;accent-color:var(--a)}}
.range-row{{display:flex;align-items:center;gap:10px}}
.range-val{{font-size:12px;color:var(--a3);min-width:36px;text-align:right}}
/* VOICE GRID */
.voice-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:4px}}
.vbtn{{background:var(--s2);border:2px solid var(--bd);color:var(--dim);padding:10px 6px;border-radius:10px;cursor:pointer;font-size:11px;text-align:center;transition:all .15s;width:100%}}
.vbtn:hover{{border-color:var(--a);color:var(--t)}}
.vbtn.sel{{border-color:var(--a);background:#7c3aed18;color:var(--a3)}}
.vbtn .vi{{font-size:22px;display:block;margin-bottom:3px}}
.vbtn .vn{{font-weight:700;display:block;font-size:12px}}
.vbtn .vd{{display:block;font-size:9px;margin-top:1px;color:var(--dim)}}
.vbtn.sel .vd{{color:var(--a3)}}
/* PILLS */
.pills{{display:flex;flex-wrap:wrap;gap:5px;margin-top:4px}}
.pill{{background:var(--s2);border:1px solid var(--bd);color:var(--dim);padding:4px 10px;border-radius:16px;cursor:pointer;font-size:11px;transition:all .15s;border:none;font-family:inherit}}
.pill:hover{{background:var(--s3);color:var(--t)}}
.pill.sel{{background:#7c3aed25;color:var(--a3);outline:1px solid var(--a)}}
/* BTN */
.btn{{background:linear-gradient(135deg,var(--a),var(--a2));color:#fff;border:none;padding:12px 20px;border-radius:10px;font-size:14px;font-weight:700;cursor:pointer;width:100%;transition:all .2s;letter-spacing:.3px;margin-top:14px}}
.btn:hover{{transform:translateY(-1px);box-shadow:0 6px 20px #7c3aed35}}
.btn:disabled{{opacity:.45;cursor:default;transform:none;box-shadow:none}}
.btn-sm{{background:var(--s2);border:1px solid var(--bd);color:var(--t);padding:7px 14px;border-radius:8px;cursor:pointer;font-size:12px;font-family:inherit;transition:all .15s}}
.btn-sm:hover{{border-color:var(--a);color:var(--a3)}}
.btn-danger{{background:#ef444415;border:1px solid #ef444440;color:var(--r)}}
/* STATUS */
.status{{padding:10px 14px;border-radius:9px;font-size:13px;display:none;align-items:center;gap:8px;margin-top:10px}}
.status.show{{display:flex}}
.status.ok{{background:#22c55e12;border:1px solid #22c55e35;color:var(--g)}}
.status.err{{background:#ef444412;border:1px solid #ef444435;color:var(--r)}}
.status.loading{{background:#7c3aed12;border:1px solid #7c3aed35;color:var(--a3)}}
.spin{{width:15px;height:15px;border:2px solid #a855f740;border-top-color:var(--a3);border-radius:50%;animation:sp .7s linear infinite;flex-shrink:0}}
@keyframes sp{{to{{transform:rotate(360deg)}}}}
/* AUDIO PLAYER */
.player-wrap{{background:var(--s2);border:1px solid var(--bd);border-radius:12px;padding:16px;margin-top:12px}}
audio{{width:100%;border-radius:8px;height:38px}}
.waveform-canvas{{width:100%;height:56px;border-radius:8px;background:var(--s3);margin-top:10px;cursor:pointer}}
.audio-meta{{display:flex;gap:16px;margin-top:8px;font-size:11px;color:var(--dim)}}
.audio-meta span{{color:var(--t)}}
/* ACTION BUTTONS */
.action-row{{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}}
/* CHAR COUNT */
.char-row{{display:flex;justify-content:space-between;font-size:11px;color:var(--dim);margin-top:4px}}
/* UPLOAD */
.upload-zone{{border:2px dashed var(--bd);border-radius:10px;padding:28px;text-align:center;cursor:pointer;position:relative;transition:all .2s}}
.upload-zone:hover,.upload-zone.drag{{border-color:var(--a);background:#7c3aed08}}
.upload-zone.has{{border-color:var(--g);background:#22c55e08}}
.upload-zone input{{position:absolute;inset:0;opacity:0;cursor:pointer}}
/* HISTORY TABLE */
.hist-item{{background:var(--s2);border:1px solid var(--bd);border-radius:10px;padding:12px 14px;display:flex;align-items:center;gap:12px;margin-bottom:8px}}
.hist-text{{flex:1;font-size:12px;color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.hist-badge{{background:var(--s3);padding:3px 8px;border-radius:12px;font-size:10px;color:var(--a3);flex-shrink:0}}
/* PROCESSING TOGGLES */
.toggle-grid{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
.toggle-item{{background:var(--s2);border:1px solid var(--bd);border-radius:9px;padding:10px 12px;cursor:pointer;display:flex;align-items:center;gap:8px;font-size:12px;transition:all .15s}}
.toggle-item.on{{border-color:var(--a);background:#7c3aed12;color:var(--a3)}}
.toggle-dot{{width:10px;height:10px;border-radius:50%;background:var(--bd);flex-shrink:0;transition:background .15s}}
.toggle-item.on .toggle-dot{{background:var(--a3)}}
/* STATS */
.stat-card{{background:var(--s2);border:1px solid var(--bd);border-radius:10px;padding:16px;text-align:center}}
.stat-num{{font-size:28px;font-weight:800;background:linear-gradient(135deg,var(--a2),var(--b));-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.stat-label{{font-size:11px;color:var(--dim);margin-top:4px}}
/* PAGE TITLE */
.page-title{{font-size:20px;font-weight:700;margin-bottom:4px}}
.page-sub{{font-size:13px;color:var(--dim);margin-bottom:20px}}
/* PODCAST SCRIPT */
.script-line{{background:var(--s2);border:1px solid var(--bd);border-radius:9px;padding:10px;margin-bottom:8px;display:flex;gap:8px;align-items:flex-start}}
.script-speaker{{flex-shrink:0}}
.script-text{{flex:1}}
/* SCROLLBAR */
::-webkit-scrollbar{{width:5px}}::-webkit-scrollbar-track{{background:transparent}}
::-webkit-scrollbar-thumb{{background:var(--bd);border-radius:3px}}
</style>
</head>
<body>
<header>
  <div><div class="logo">QWN3-TTS Studio</div><div class="tagline">50 features · Melhor que ElevenLabs · 100% grátis</div></div>
  <div class="badges">
    <div class="badge on" id="gpu-badge">GPU: {GPU_NAME} {GPU_VRAM}</div>
    <div class="badge" id="model-badge">Carregando...</div>
  </div>
</header>
<div class="app">
  <nav class="sidebar">
    <div class="nav-section">Geração</div>
    <button class="nav-btn active" onclick="nav('generate',this)"><span class="ico">🎙️</span>Voz Pronta</button>
    <button class="nav-btn" onclick="nav('design',this)"><span class="ico">🎨</span>Criar Voz</button>
    <button class="nav-btn" onclick="nav('clone',this)"><span class="ico">🔬</span>Clonar Voz</button>
    <button class="nav-btn" onclick="nav('podcast',this)"><span class="ico">🎙</span>Podcast</button>
    <button class="nav-btn" onclick="nav('batch',this)"><span class="ico">⚡</span>Batch</button>
    <button class="nav-btn" onclick="nav('compare',this)"><span class="ico">⚖️</span>Comparar Vozes</button>
    <div class="nav-section">Ferramentas</div>
    <button class="nav-btn" onclick="nav('history',this)"><span class="ico">📋</span>Histórico</button>
    <button class="nav-btn" onclick="nav('projects',this)"><span class="ico">📁</span>Projetos</button>
    <button class="nav-btn" onclick="nav('pronunciation',this)"><span class="ico">📖</span>Pronúncia</button>
    <button class="nav-btn" onclick="nav('stats',this)"><span class="ico">📊</span>Estatísticas</button>
    <button class="nav-btn" onclick="nav('api',this)"><span class="ico">🔌</span>API</button>
  </nav>
  <div class="main">

    <!-- GENERATE -->
    <div id="page-generate" class="page active">
      <div class="page-title">🎙️ Voz Pronta</div>
      <div class="page-sub">Escolha uma voz e gere áudio com controle de estilo e emoção</div>
      <div class="grid-2">
        <div>
          <div class="card">
            <div class="card-title">🎭 Voz</div>
            <div class="voice-grid" id="vg"></div>
            <label style="margin-top:12px">Idioma</label>
            <select id="idioma1"></select>
            <label>Estilo / Emoção</label>
            <div class="pills" id="ep"></div>
            <input type="text" id="estilo-livre" placeholder="Ou descreva livremente..." style="margin-top:8px">
          </div>
          <div class="card" style="margin-top:16px">
            <div class="card-title">⚙️ Processamento de Áudio</div>
            <label>Velocidade</label>
            <div class="range-row"><input type="range" id="speed" min="0.5" max="2.0" step="0.05" value="1.0" oninput="rv('speed-v',this.value+'x')"><span class="range-val" id="speed-v">1.0x</span></div>
            <label>Tom (semitones)</label>
            <div class="range-row"><input type="range" id="pitch" min="-6" max="6" step="0.5" value="0" oninput="rv('pitch-v',this.value)"><span class="range-val" id="pitch-v">0</span></div>
            <label>EQ Preset</label>
            <select id="eq-preset"></select>
            <div style="margin-top:12px" class="toggle-grid" id="proc-toggles">
              <div class="toggle-item" onclick="tog(this,'normalize')"><div class="toggle-dot"></div>Normalizar</div>
              <div class="toggle-item" onclick="tog(this,'trim')"><div class="toggle-dot"></div>Cortar Silêncio</div>
              <div class="toggle-item" onclick="tog(this,'reverb')"><div class="toggle-dot"></div>Reverb</div>
              <div class="toggle-item" onclick="tog(this,'echo')"><div class="toggle-dot"></div>Echo</div>
              <div class="toggle-item" onclick="tog(this,'compress')"><div class="toggle-dot"></div>Compressão</div>
              <div class="toggle-item" onclick="tog(this,'fade')"><div class="toggle-dot"></div>Fade In/Out</div>
              <div class="toggle-item" onclick="tog(this,'padding')"><div class="toggle-dot"></div>Padding</div>
              <div class="toggle-item" onclick="tog(this,'auto_punctuation')"><div class="toggle-dot"></div>Auto Pontuação</div>
            </div>
            <label style="margin-top:12px">Exportar Formatos</label>
            <div class="toggle-grid">
              <div class="toggle-item" onclick="tog(this,'export_mp3')"><div class="toggle-dot"></div>MP3</div>
              <div class="toggle-item" onclick="tog(this,'generate_srt')"><div class="toggle-dot"></div>Legendas SRT</div>
            </div>
          </div>
        </div>
        <div>
          <div class="card">
            <div class="card-title">📝 Texto</div>
            <textarea id="texto1" placeholder="Escreva aqui o que a IA vai falar..." oninput="textStats(this,'stats1','count1')"></textarea>
            <div class="char-row">
              <span id="stats1" style="color:var(--a3)"></span>
              <span><span id="count1">0</span> chars</span>
            </div>
            <div style="font-size:11px;color:var(--dim);margin-top:6px">
              Suporte a tags: &lt;break/&gt; &lt;emphasis&gt;texto&lt;/emphasis&gt; &lt;slow&gt;texto&lt;/slow&gt;
            </div>
            <button class="btn" id="btn1" onclick="generate()">⚡ GERAR ÁUDIO</button>
          </div>
          <div id="status1" class="status"></div>
          <div id="result1" style="display:none">
            <div class="player-wrap">
              <audio id="audio1" controls></audio>
              <canvas id="wave-canvas" class="waveform-canvas"></canvas>
              <div class="audio-meta">Duração: <span id="r-dur">-</span> &nbsp; Peak: <span id="r-peak">-</span> &nbsp; RMS: <span id="r-rms">-</span></div>
            </div>
            <div class="action-row">
              <a id="dl1" class="btn-sm" download>💾 WAV</a>
              <a id="dl-mp3" class="btn-sm" style="display:none" download>🎵 MP3</a>
              <a id="dl-srt" class="btn-sm" style="display:none" download>📄 SRT</a>
              <button class="btn-sm" onclick="copyText('srt-content')">📋 Copiar SRT</button>
              <button class="btn-sm" onclick="showCost()">💰 Custo Salvo</button>
            </div>
            <div id="cost-box" style="display:none;margin-top:8px;padding:10px;background:var(--s2);border-radius:9px;font-size:12px"></div>
            <div id="srt-content" style="display:none"></div>
          </div>
        </div>
      </div>
    </div>

    <!-- DESIGN -->
    <div id="page-design" class="page">
      <div class="page-title">🎨 Criar Voz</div>
      <div class="page-sub">Descreva qualquer voz e a IA cria para você</div>
      <div class="grid-2">
        <div class="card">
          <label>Idioma</label><select id="idioma2"></select>
          <label>Descrição da voz</label>
          <textarea id="descricao2" rows="3" placeholder='"voz masculina grave e calma, como locutor de rádio dos anos 80"&#10;"voz feminina jovem animada, como apresentadora de TV"'></textarea>
          <label>Texto</label>
          <textarea id="texto2" placeholder="Escreva o que essa voz vai falar..." oninput="cnt(this,'cnt2')"></textarea>
          <div class="char-row"><span></span><span id="cnt2">0</span></div>
          <button class="btn" id="btn2" onclick="genDesign()">🎨 CRIAR E GERAR</button>
        </div>
        <div>
          <div id="status2" class="status"></div>
          <div id="result2" style="display:none" class="player-wrap">
            <audio id="audio2" controls></audio>
            <div class="action-row"><a id="dl2" class="btn-sm" download>💾 Baixar WAV</a></div>
          </div>
        </div>
      </div>
    </div>

    <!-- CLONE -->
    <div id="page-clone" class="page">
      <div class="page-title">🔬 Clonar Voz</div>
      <div class="page-sub">Clone qualquer voz a partir de um áudio de referência</div>
      <div class="grid-2">
        <div class="card">
          <label>Idioma</label><select id="idioma3"></select>
          <label>Áudio de referência</label>
          <div class="upload-zone" id="uz"><input type="file" id="aref" accept="audio/*" onchange="onFile(this)"><span style="font-size:28px">🎤</span><p id="up-txt" style="margin-top:8px;color:var(--dim);font-size:13px">Clique ou arraste um áudio<br><small>MP3, WAV, M4A — até 30MB</small></p></div>
          <label>Transcrição (opcional)</label>
          <textarea id="trans" rows="2" placeholder='O que o áudio diz? Ex: "Olá, sou o João..."'></textarea>
          <label>Novo texto</label>
          <textarea id="texto3" placeholder="Será falado com a voz clonada..." oninput="cnt(this,'cnt3')"></textarea>
          <div class="char-row"><span></span><span id="cnt3">0</span></div>
          <button class="btn" id="btn3" onclick="genClone()">🔬 CLONAR E GERAR</button>
        </div>
        <div>
          <div id="status3" class="status"></div>
          <div id="result3" style="display:none" class="player-wrap">
            <audio id="audio3" controls></audio>
            <div class="action-row"><a id="dl3" class="btn-sm" download>💾 Baixar WAV</a></div>
          </div>
        </div>
      </div>
    </div>

    <!-- PODCAST -->
    <div id="page-podcast" class="page">
      <div class="page-title">🎙 Modo Podcast</div>
      <div class="page-sub">Conversas multi-speaker com vozes diferentes</div>
      <div class="grid-2">
        <div class="card">
          <div class="card-title">Script</div>
          <div id="script-lines"></div>
          <button class="btn-sm" onclick="addScriptLine()" style="margin-top:8px;width:100%">+ Adicionar fala</button>
          <button class="btn" onclick="genPodcast()" id="btn-pod">🎙 GERAR PODCAST</button>
        </div>
        <div>
          <div id="status-pod" class="status"></div>
          <div id="result-pod" style="display:none" class="player-wrap">
            <audio id="audio-pod" controls></audio>
            <div class="action-row"><a id="dl-pod" class="btn-sm" download>💾 Baixar WAV</a></div>
          </div>
        </div>
      </div>
    </div>

    <!-- BATCH -->
    <div id="page-batch" class="page">
      <div class="page-title">⚡ Batch — Geração em Massa</div>
      <div class="page-sub">Gere múltiplos áudios de uma vez (um texto por linha)</div>
      <div class="grid-2">
        <div class="card">
          <label>Voz</label><select id="batch-voz"></select>
          <label>Idioma</label><select id="batch-idioma"></select>
          <label>Textos (um por linha)</label>
          <textarea id="batch-texts" rows="10" placeholder="Primeira frase aqui&#10;Segunda frase aqui&#10;Terceira frase aqui"></textarea>
          <button class="btn" id="btn-batch" onclick="genBatch()">⚡ GERAR TUDO</button>
        </div>
        <div class="card">
          <div class="card-title">Resultados</div>
          <div id="batch-results"></div>
        </div>
      </div>
    </div>

    <!-- COMPARE -->
    <div id="page-compare" class="page">
      <div class="page-title">⚖️ Comparar Vozes</div>
      <div class="page-sub">Teste o mesmo texto com duas vozes lado a lado</div>
      <div class="card" style="max-width:600px">
        <div class="grid-2">
          <div><label>Voz A</label><select id="cmp-a"></select></div>
          <div><label>Voz B</label><select id="cmp-b"></select></div>
        </div>
        <label>Idioma</label><select id="cmp-idioma"></select>
        <label>Texto</label>
        <textarea id="cmp-text" placeholder="Mesmo texto será gerado com as duas vozes..."></textarea>
        <button class="btn" onclick="genCompare()">⚖️ COMPARAR</button>
      </div>
      <div id="cmp-results" class="grid-2" style="margin-top:16px"></div>
    </div>

    <!-- HISTORY -->
    <div id="page-history" class="page">
      <div class="page-title">📋 Histórico</div>
      <div class="page-sub">Todos os áudios gerados nesta sessão</div>
      <div style="display:flex;gap:8px;margin-bottom:16px">
        <input type="text" id="hist-search" placeholder="Buscar..." oninput="searchHist(this.value)" style="flex:1">
        <button class="btn-sm" onclick="exportHistory()">📥 Exportar CSV</button>
      </div>
      <div id="hist-list"></div>
    </div>

    <!-- PROJECTS -->
    <div id="page-projects" class="page">
      <div class="page-title">📁 Projetos</div>
      <div class="page-sub">Salve e carregue configurações de projetos</div>
      <div class="grid-2">
        <div class="card">
          <div class="card-title">Salvar Projeto Atual</div>
          <label>Nome do projeto</label>
          <input type="text" id="proj-name" placeholder="Meu Podcast...">
          <button class="btn" onclick="saveProject()">💾 Salvar</button>
        </div>
        <div class="card">
          <div class="card-title">Projetos Salvos</div>
          <div id="proj-list"></div>
        </div>
      </div>
    </div>

    <!-- PRONUNCIATION -->
    <div id="page-pronunciation" class="page">
      <div class="page-title">📖 Dicionário de Pronúncia</div>
      <div class="page-sub">Ensine a IA como pronunciar palavras específicas</div>
      <div class="grid-2">
        <div class="card">
          <label>Palavra</label><input type="text" id="pron-word" placeholder="João">
          <label>Pronunciar como</label><input type="text" id="pron-as" placeholder="Joãw">
          <button class="btn" onclick="addPron()">+ Adicionar</button>
        </div>
        <div class="card">
          <div class="card-title">Dicionário atual</div>
          <div id="pron-list"></div>
        </div>
      </div>
    </div>

    <!-- STATS -->
    <div id="page-stats" class="page">
      <div class="page-title">📊 Estatísticas</div>
      <div class="page-sub">Uso do estúdio e economia vs ElevenLabs</div>
      <div class="grid-3" id="stats-cards" style="margin-bottom:16px"></div>
      <div class="card">
        <div class="card-title">💰 Economia vs ElevenLabs</div>
        <div id="cost-stats"></div>
      </div>
    </div>

    <!-- API -->
    <div id="page-api" class="page">
      <div class="page-title">🔌 API REST</div>
      <div class="page-sub">Use o QWN3-TTS em qualquer aplicação</div>
      <div class="card">
        <div class="card-title">Endpoints disponíveis</div>
        <pre style="background:var(--s2);padding:16px;border-radius:10px;font-size:12px;overflow-x:auto;color:var(--a3)">POST /api/generate        — Gerar com voz pronta
POST /api/generate/design — Criar voz nova
POST /api/generate/clone  — Clonar voz
POST /api/batch           — Batch múltiplos textos
POST /api/podcast         — Modo podcast multi-speaker
POST /api/compare         — Comparar duas vozes
GET  /api/history         — Histórico
GET  /api/stats           — Estatísticas
GET  /api/audio/{{file}}   — Servir arquivo de áudio
POST /api/keys            — Gerar API key</pre>
        <label style="margin-top:16px">Gerar API Key</label>
        <div style="display:flex;gap:8px">
          <input type="text" id="key-name" placeholder="Nome da aplicação" style="flex:1">
          <button class="btn-sm" onclick="genKey()">Gerar</button>
        </div>
        <div id="key-result" style="margin-top:8px;font-size:12px;color:var(--a3)"></div>
        <label style="margin-top:16px">Exemplo cURL</label>
        <pre style="background:var(--s2);padding:12px;border-radius:9px;font-size:11px;overflow-x:auto">curl -X POST http://localhost:7860/api/generate \\
  -H "Content-Type: application/json" \\
  -d '{{"texto":"Olá mundo!","voz":"Ryan","idioma":"Portuguese"}}'</pre>
      </div>
    </div>

  </div>
</div>
<script>
const VOZES={vozes_j};
const IDIOMAS={idiomas_j};
const ESTILOS={estilos_j};
const EQ_PRESETS={eq_j};
let selVoice=Object.keys(VOZES)[0], selStyle='', toggles={{}}, lastResult=null;

// ── INIT ─────────────────────────────────────────────────────────────────────
window.onload=async()=>{{
  // GPU info
  try{{const i=await fetch('/api/info').then(r=>r.json());
    document.getElementById('model-badge').textContent=i.models_loaded.length+'/3 modelos';}}catch{{}}
  // Voice grid
  const vg=document.getElementById('vg');
  Object.entries(VOZES).forEach(([k,v])=>{{
    const b=document.createElement('button');b.className='vbtn'+(k===selVoice?' sel':'');
    b.innerHTML=`<span class="vi">${{v.icon}}</span><span class="vn">${{v.label}}</span><span class="vd">${{v.desc}}</span>`;
    b.onclick=()=>{{document.querySelectorAll('.vbtn').forEach(x=>x.classList.remove('sel'));b.classList.add('sel');selVoice=k;}};
    vg.appendChild(b);
  }});
  // Idiomas
  ['idioma1','idioma2','idioma3','batch-idioma','cmp-idioma'].forEach(id=>{{
    const s=document.getElementById(id);if(!s)return;
    Object.keys(IDIOMAS).forEach(l=>{{const o=document.createElement('option');o.value=l;o.textContent=l;if(l==='Português')o.selected=true;s.appendChild(o);}});
  }});
  // Batch voz + compare selects
  ['batch-voz','cmp-a','cmp-b'].forEach(id=>{{
    const s=document.getElementById(id);if(!s)return;
    Object.entries(VOZES).forEach(([k,v])=>{{const o=document.createElement('option');o.value=k;o.textContent=v.label;s.appendChild(o);}});
  }});
  if(document.getElementById('cmp-b')) document.getElementById('cmp-b').value='Vivian';
  // Style pills
  const ep=document.getElementById('ep');
  ESTILOS.forEach(s=>{{
    const p=document.createElement('button');p.className='pill'+(s===''?' sel':'');
    p.textContent=s||'Normal';p.onclick=()=>{{document.querySelectorAll('.pill').forEach(x=>x.classList.remove('sel'));p.classList.add('sel');selStyle=s;document.getElementById('estilo-livre').value='';}};
    ep.appendChild(p);
  }});
  // EQ
  const eq=document.getElementById('eq-preset');
  EQ_PRESETS.forEach(e=>{{const o=document.createElement('option');o.value=e;o.textContent=e;eq.appendChild(o);}});
  // Podcast default lines
  addScriptLine(); addScriptLine();
  // Load history
  loadHistory(); loadProjects(); loadPron(); loadStats();
  // Drag and drop upload
  const uz=document.getElementById('uz');
  uz.addEventListener('dragover',e=>{{e.preventDefault();uz.classList.add('drag')}});
  uz.addEventListener('dragleave',()=>uz.classList.remove('drag'));
  uz.addEventListener('drop',e=>{{e.preventDefault();uz.classList.remove('drag');const f=e.dataTransfer.files[0];if(f){{document.getElementById('aref').files=e.dataTransfer.files;onFile({{files:[f]}});}}}});
}};

// ── NAV ──────────────────────────────────────────────────────────────────────
function nav(page,el){{
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));
  document.getElementById('page-'+page).classList.add('active');
  el.classList.add('active');
  if(page==='history')loadHistory();
  if(page==='stats')loadStats();
  if(page==='projects')loadProjects();
  if(page==='pronunciation')loadPron();
}}

// ── HELPERS ──────────────────────────────────────────────────────────────────
function rv(id,v){{document.getElementById(id).textContent=v;}}
function cnt(el,id){{document.getElementById(id).textContent=el.value.length;}}
function tog(el,key){{el.classList.toggle('on');toggles[key]=el.classList.contains('on');}}
function setStatus(id,type,msg){{const e=document.getElementById(id);e.className='status show '+type;e.innerHTML=type==='loading'?`<div class="spin"></div>${{msg}}`:msg;}}
function hideStatus(id){{document.getElementById(id).className='status';}}
function b64Blob(b64){{const bin=atob(b64);const a=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)a[i]=bin.charCodeAt(i);return new Blob([a],{{type:'audio/wav'}});}}
function textStats(el,sid,cid){{
  const w=el.value.trim().split(/\s+/).filter(Boolean).length;
  const c=el.value.length;
  const est=Math.round(w/2.8);
  document.getElementById(sid).textContent=`${{w}} palavras · ~${{Math.floor(est/60)}}:${{String(est%60).padStart(2,'0')}}`;
  document.getElementById(cid).textContent=c;
}}
function copyText(id){{const e=document.getElementById(id);if(e){{navigator.clipboard.writeText(e.textContent||e.innerText);}}}}

// ── WAVEFORM ─────────────────────────────────────────────────────────────────
function drawWaveform(data){{
  const canvas=document.getElementById('wave-canvas');
  if(!canvas||!data)return;
  const ctx=canvas.getContext('2d');
  canvas.width=canvas.offsetWidth*2;canvas.height=canvas.offsetHeight*2;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const grad=ctx.createLinearGradient(0,0,canvas.width,0);
  grad.addColorStop(0,'#7c3aed');grad.addColorStop(0.5,'#a855f7');grad.addColorStop(1,'#60a5fa');
  ctx.fillStyle=grad;
  const barW=canvas.width/data.length;
  const mid=canvas.height/2;
  data.forEach((v,i)=>{{
    const h=Math.max(2,v*mid*0.9);
    ctx.fillRect(i*barW,mid-h,barW*0.7,h*2);
  }});
}}

// ── GENERATE ─────────────────────────────────────────────────────────────────
async function generate(){{
  const texto=document.getElementById('texto1').value.trim();
  if(!texto)return;
  const estilo=document.getElementById('estilo-livre').value.trim()||selStyle||null;
  const idioma=IDIOMAS[document.getElementById('idioma1').value];
  const opts={{...toggles,speed:parseFloat(document.getElementById('speed').value),pitch:parseFloat(document.getElementById('pitch').value),eq:document.getElementById('eq-preset').value}};
  document.getElementById('btn1').disabled=true;
  document.getElementById('result1').style.display='none';
  setStatus('status1','loading','⚡ Gerando áudio com 50 features...');
  try{{
    const d=await fetch('/api/generate',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{texto,voz:selVoice,idioma,estilo,opts}})}}).then(r=>r.json());
    if(d.error){{setStatus('status1','err','❌ '+d.error);return;}}
    lastResult=d;
    const url=URL.createObjectURL(b64Blob(d.audio_b64));
    document.getElementById('audio1').src=url;
    document.getElementById('dl1').href=url;document.getElementById('dl1').download=d.filename;
    document.getElementById('r-dur').textContent=d.stats.duration_s+'s';
    document.getElementById('r-peak').textContent=d.stats.peak_db+'dB';
    document.getElementById('r-rms').textContent=d.stats.rms_db+'dB';
    document.getElementById('result1').style.display='block';
    drawWaveform(d.waveform);
    if(d.exports&&d.exports.mp3){{const el=document.getElementById('dl-mp3');el.href='/api/audio/'+d.exports.mp3.split('\\\\').pop().split('/').pop();el.style.display='flex';}}
    if(d.srt){{document.getElementById('dl-srt').style.display='flex';document.getElementById('srt-content').textContent=d.srt;}}
    setStatus('status1','ok',`✅ Gerado em ${{d.duration_s.toFixed(1)}}s · ${{d.stats.duration_s}}s de áudio · Salvo: ${{d.filename}}`);
  }}catch(e){{setStatus('status1','err','❌ '+e.message);}}
  finally{{document.getElementById('btn1').disabled=false;}}
}}

function showCost(){{
  if(!lastResult)return;
  const c=lastResult.cost_saved;
  const el=document.getElementById('cost-box');
  el.style.display='block';
  el.innerHTML=`💰 <b>Economia:</b> ElevenLabs cobraria <b>${{c.eleven_labs_usd}}</b> por ${{c.chars}} caracteres. <br>Aqui: <b style="color:var(--g)">$0.00</b> — você economizou ${{c.savings_usd}} (100%)`;
}}

// ── DESIGN ───────────────────────────────────────────────────────────────────
async function genDesign(){{
  const texto=document.getElementById('texto2').value.trim();
  const desc=document.getElementById('descricao2').value.trim();
  if(!texto||!desc)return;
  document.getElementById('btn2').disabled=true;
  setStatus('status2','loading','🎨 Criando voz...');
  try{{
    const d=await fetch('/api/generate/design',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{texto,idioma:IDIOMAS[document.getElementById('idioma2').value],descricao:desc}})}}).then(r=>r.json());
    if(d.error){{setStatus('status2','err','❌ '+d.error);return;}}
    const url=URL.createObjectURL(b64Blob(d.audio_b64));
    document.getElementById('audio2').src=url;document.getElementById('dl2').href=url;document.getElementById('dl2').download=d.filename;
    document.getElementById('result2').style.display='block';
    setStatus('status2','ok','✅ Voz criada em '+d.duration_s.toFixed(1)+'s');
  }}catch(e){{setStatus('status2','err','❌ '+e.message);}}
  finally{{document.getElementById('btn2').disabled=false;}}
}}

// ── CLONE ────────────────────────────────────────────────────────────────────
async function genClone(){{
  const file=document.getElementById('aref').files[0];
  if(!file)return;
  document.getElementById('btn3').disabled=true;
  setStatus('status3','loading','🔬 Clonando voz...');
  const fd=new FormData();
  fd.append('audio_ref',file);fd.append('texto',document.getElementById('texto3').value);
  fd.append('idioma',IDIOMAS[document.getElementById('idioma3').value]);
  fd.append('transcricao',document.getElementById('trans').value);
  try{{
    const d=await fetch('/api/generate/clone',{{method:'POST',body:fd}}).then(r=>r.json());
    if(d.error){{setStatus('status3','err','❌ '+d.error);return;}}
    const url=URL.createObjectURL(b64Blob(d.audio_b64));
    document.getElementById('audio3').src=url;document.getElementById('dl3').href=url;document.getElementById('dl3').download=d.filename;
    document.getElementById('result3').style.display='block';
    setStatus('status3','ok','✅ Clone gerado em '+d.duration_s.toFixed(1)+'s');
  }}catch(e){{setStatus('status3','err','❌ '+e.message);}}
  finally{{document.getElementById('btn3').disabled=false;}}
}}

function onFile(input){{
  const f=input.files?.[0]||input;if(!f)return;
  document.getElementById('uz').classList.add('has');
  document.getElementById('up-txt').textContent='✅ '+f.name;
}}

// ── PODCAST ──────────────────────────────────────────────────────────────────
let scriptLines=[];
function addScriptLine(){{
  const i=scriptLines.length;
  scriptLines.push({{speaker:'Ryan',text:''}});
  const div=document.getElementById('script-lines');
  const row=document.createElement('div');row.className='script-line';row.id='sl-'+i;
  row.innerHTML=`<div class="script-speaker"><select onchange="scriptLines[${{i}}].speaker=this.value" style="width:120px">${{Object.entries(VOZES).map(([k,v])=>`<option value="${{k}}">${{v.label}}</option>`).join('')}}</select></div><div class="script-text"><textarea rows="2" placeholder="Fala desta pessoa..." oninput="scriptLines[${{i}}].text=this.value" style="min-height:60px"></textarea></div><button class="btn-sm btn-danger" onclick="removeScriptLine(${{i}})" style="flex-shrink:0">✕</button>`;
  div.appendChild(row);
}}
function removeScriptLine(i){{document.getElementById('sl-'+i)?.remove();scriptLines[i]=null;}}
async function genPodcast(){{
  const lines=scriptLines.filter(Boolean).filter(l=>l.text.trim());
  if(!lines.length)return;
  document.getElementById('btn-pod').disabled=true;
  setStatus('status-pod','loading','🎙 Gerando podcast...');
  try{{
    const d=await fetch('/api/podcast',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{script:lines}})}}).then(r=>r.json());
    if(d.error){{setStatus('status-pod','err','❌ '+d.error);return;}}
    const url=URL.createObjectURL(b64Blob(d.audio_b64));
    document.getElementById('audio-pod').src=url;document.getElementById('dl-pod').href=url;document.getElementById('dl-pod').download=d.filename;
    document.getElementById('result-pod').style.display='block';
    setStatus('status-pod','ok','✅ Podcast gerado!');
  }}catch(e){{setStatus('status-pod','err','❌ '+e.message);}}
  finally{{document.getElementById('btn-pod').disabled=false;}}
}}

// ── BATCH ────────────────────────────────────────────────────────────────────
async function genBatch(){{
  const texts=document.getElementById('batch-texts').value.split('\\n').filter(t=>t.trim());
  if(!texts.length)return;
  document.getElementById('btn-batch').disabled=true;
  document.getElementById('batch-results').innerHTML='<div style="color:var(--dim)">⏳ Gerando '+texts.length+' áudios...</div>';
  try{{
    const d=await fetch('/api/batch',{{method:'POST',headers:{{'Content-Type':'application/json'}},
      body:JSON.stringify({{texts,voz:document.getElementById('batch-voz').value,idioma:IDIOMAS[document.getElementById('batch-idioma').value]}})}}).then(r=>r.json());
    const div=document.getElementById('batch-results');div.innerHTML='';
    d.results.forEach((r,i)=>{{
      const el=document.createElement('div');el.className='hist-item';
      el.innerHTML=r.ok?`<span style="color:var(--g)">✓</span><span class="hist-text">${{r.text}}</span><a class="btn-sm" href="/api/audio/${{r.path.split(/[\\/\\\\]/).pop()}}" download>💾</a>`:`<span style="color:var(--r)">✗</span><span class="hist-text">${{r.text}} — ${{r.error}}</span>`;
      div.appendChild(el);
    }});
  }}catch(e){{document.getElementById('batch-results').innerHTML='<div style="color:var(--r)">Erro: '+e.message+'</div>';}}
  finally{{document.getElementById('btn-batch').disabled=false;}}
}}

// ── COMPARE ──────────────────────────────────────────────────────────────────
async function genCompare(){{
  const texto=document.getElementById('cmp-text').value.trim();if(!texto)return;
  const va=document.getElementById('cmp-a').value,vb=document.getElementById('cmp-b').value;
  const idioma=IDIOMAS[document.getElementById('cmp-idioma').value];
  const div=document.getElementById('cmp-results');div.innerHTML='<div style="color:var(--dim)">⏳ Gerando comparação...</div>';
  try{{
    const d=await fetch('/api/compare',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{texto,voice_a:va,voice_b:vb,idioma}})}}).then(r=>r.json());
    div.innerHTML='';
    Object.entries(d).forEach(([voz,data])=>{{
      const url=URL.createObjectURL(b64Blob(data.b64));
      const c=document.createElement('div');c.className='player-wrap';
      c.innerHTML=`<div style="font-weight:700;margin-bottom:8px">${{VOZES[voz]?.icon}} ${{VOZES[voz]?.label||voz}}</div><audio src="${{url}}" controls></audio>`;
      div.appendChild(c);
    }});
  }}catch(e){{div.innerHTML='<div style="color:var(--r)">Erro: '+e.message+'</div>';}}
}}

// ── HISTORY ──────────────────────────────────────────────────────────────────
async function loadHistory(q=''){{
  const url=q?`/api/history?q=${{encodeURIComponent(q)}}`:'/api/history';
  const data=await fetch(url).then(r=>r.json()).catch(()=>[]);
  const div=document.getElementById('hist-list');div.innerHTML='';
  if(!data.length){{div.innerHTML='<div style="color:var(--dim);font-size:13px">Nenhum áudio ainda.</div>';return;}}
  data.forEach(h=>{{
    const el=document.createElement('div');el.className='hist-item';
    const file=h.path?.split(/[\\/\\\\]/).pop()||'';
    el.innerHTML=`<span class="hist-badge">${{h.voice||'?'}}</span><span class="hist-text">${{h.text?.substring(0,80)||''}}</span><span style="font-size:10px;color:var(--dim);flex-shrink:0">${{h.duration||0}}s</span>${{file?`<a class="btn-sm" href="/api/audio/${{file}}" download>💾</a>`:''}}`;
    div.appendChild(el);
  }});
}}
function searchHist(q){{loadHistory(q);}}
async function exportHistory(){{window.open('/api/history/export');}}

// ── PROJECTS ─────────────────────────────────────────────────────────────────
async function loadProjects(){{
  const data=await fetch('/api/projects').then(r=>r.json()).catch(()=>[]);
  const div=document.getElementById('proj-list');div.innerHTML='';
  if(!data.length){{div.innerHTML='<div style="color:var(--dim);font-size:12px">Nenhum projeto salvo.</div>';return;}}
  data.forEach(name=>{{const el=document.createElement('div');el.style.cssText='display:flex;align-items:center;gap:8px;margin-bottom:6px';el.innerHTML=`<span style="flex:1;font-size:13px">📁 ${{name}}</span><button class="btn-sm" onclick="loadProject('${{name}}')">Carregar</button>`;div.appendChild(el);}});
}}
async function saveProject(){{
  const name=document.getElementById('proj-name').value.trim();if(!name)return;
  const data={{text:document.getElementById('texto1').value,voice:selVoice,style:selStyle}};
  await fetch('/api/projects',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{name,data}})}});
  loadProjects();
}}
async function loadProject(name){{
  const d=await fetch(`/api/projects/${{name}}`).then(r=>r.json());
  if(d.text)document.getElementById('texto1').value=d.text;
}}

// ── PRONUNCIATION ────────────────────────────────────────────────────────────
async function loadPron(){{
  const data=await fetch('/api/pronunciations').then(r=>r.json()).catch(()=>({{}}));
  const div=document.getElementById('pron-list');div.innerHTML='';
  if(!Object.keys(data).length){{div.innerHTML='<div style="color:var(--dim);font-size:12px">Dicionário vazio.</div>';return;}}
  Object.entries(data).forEach(([w,p])=>{{const el=document.createElement('div');el.style.cssText='display:flex;align-items:center;gap:8px;margin-bottom:6px';el.innerHTML=`<span style="flex:1;font-size:13px"><b>${{w}}</b> → ${{p}}</span><button class="btn-sm btn-danger" onclick="delPron('${{w}}')">✕</button>`;div.appendChild(el);}});
}}
async function addPron(){{
  const w=document.getElementById('pron-word').value.trim(),p=document.getElementById('pron-as').value.trim();if(!w||!p)return;
  await fetch('/api/pronunciations',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{word:w,phonetic:p}})}});
  document.getElementById('pron-word').value='';document.getElementById('pron-as').value='';loadPron();
}}
async function delPron(w){{await fetch(`/api/pronunciations/${{encodeURIComponent(w)}}`,{{method:'DELETE'}});loadPron();}}

// ── STATS ────────────────────────────────────────────────────────────────────
async function loadStats(){{
  const d=await fetch('/api/stats').then(r=>r.json()).catch(()=>({{}}));
  const cards=[
    {{n:d.total_generated||0,l:'Áudios Gerados'}},
    {{n:d.total_chars||0,l:'Caracteres Processados'}},
    {{n:(d.total_seconds||0)+'s',l:'Áudio Gerado'}},
    {{n:d.history_count||0,l:'No Histórico'}},
    {{n:d.projects||0,l:'Projetos Salvos'}},
    {{n:'$0.00',l:'Custo Total'}},
  ];
  const div=document.getElementById('stats-cards');div.innerHTML='';
  cards.forEach(c=>{{div.innerHTML+=`<div class="stat-card"><div class="stat-num">${{c.n}}</div><div class="stat-label">${{c.l}}</div></div>`;}});
  const chars=d.total_chars||0;
  const cost=(chars/1000*0.30).toFixed(4);
  document.getElementById('cost-stats').innerHTML=`<div style="font-size:13px;line-height:2">ElevenLabs cobraria: <b style="color:var(--r)">${{cost}}</b><br>QWN3-TTS: <b style="color:var(--g)">$0.00</b><br>Economia total: <b style="color:var(--a3)">${{cost}} (100%)</b></div>`;
}}

// ── API KEY ──────────────────────────────────────────────────────────────────
async function genKey(){{
  const name=document.getElementById('key-name').value.trim()||'default';
  const d=await fetch('/api/keys',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{name}})}}).then(r=>r.json());
  document.getElementById('key-result').textContent='🔑 '+d.key;
}}
</script>
</body>
</html>""".replace("{vozes_j}",vozes_j).replace("{idiomas_j}",idiomas_j).replace("{estilos_j}",estilos_j).replace("{eq_j}",eq_j)

if __name__=="__main__":
    print("\n"+"="*55)
    print("  QWN3-TTS STUDIO v3.0 — 50 Features Edition")
    print(f"  GPU: {GPU_NAME} {GPU_VRAM}")
    print("  Melhor que ElevenLabs · Grátis · Local")
    print("  http://localhost:7860")
    print("="*55+"\n")
    uvicorn.run(app,host="0.0.0.0",port=7860,log_level="warning")

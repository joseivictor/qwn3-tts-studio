"""
QWN3-TTS STUDIO v5.0 — Multi-Engine Production Edition
4 engines: Kokoro · F5-TTS · Chatterbox · Edge TTS
Run: tts_env/Scripts/python.exe -X utf8 tts_studio.py
"""

import io, os, sys, json, re, csv, uuid, time, base64, threading, tempfile, hashlib, asyncio
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Dirs ──────────────────────────────────────────────────────────────────────
OUT_DIR    = Path(__file__).parent / "audio_output"
PROJ_DIR   = Path(__file__).parent / "tts_projects"
HIST_FILE  = Path(__file__).parent / "tts_history.json"
PRON_FILE  = Path(__file__).parent / "tts_pronunciations.json"
VOICES_FILE= Path(__file__).parent / "tts_saved_voices.json"
for d in [OUT_DIR, PROJ_DIR]: d.mkdir(exist_ok=True)

# ── GPU ───────────────────────────────────────────────────────────────────────
try:
    import torch
    CUDA     = torch.cuda.is_available()
    DEVICE   = "cuda" if CUDA else "cpu"
    GPU_NAME = torch.cuda.get_device_name(0) if CUDA else "CPU"
    GPU_VRAM = f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB" if CUDA else ""
except ImportError:
    CUDA = False; DEVICE = "cpu"; GPU_NAME = "CPU"; GPU_VRAM = ""

# ── Audio libs ────────────────────────────────────────────────────────────────
try:
    import librosa; LIBROSA_OK = True
except ImportError:
    LIBROSA_OK = False

try:
    from pydub import AudioSegment; PYDUB_OK = True
except ImportError:
    PYDUB_OK = False

# ── Engine registry ───────────────────────────────────────────────────────────
_engines     = {}
_engine_lock = threading.Lock()

def _load_kokoro():
    with _engine_lock:
        if "kokoro" not in _engines:
            from kokoro import KPipeline
            _engines["kokoro"] = {
                "a": KPipeline(lang_code="a"),   # American EN
                "b": KPipeline(lang_code="b"),   # British EN
                "p": KPipeline(lang_code="p"),   # PT-BR
                "e": KPipeline(lang_code="e"),   # Spanish (es)
                "f": KPipeline(lang_code="f"),   # French
            }
    return _engines["kokoro"]

def _load_f5():
    with _engine_lock:
        if "f5" not in _engines:
            from f5_tts.api import F5TTS
            _engines["f5"] = F5TTS(device=DEVICE)
    return _engines["f5"]

def _load_chatterbox():
    with _engine_lock:
        if "chatterbox" not in _engines:
            from chatterbox.tts import ChatterboxTTS
            _engines["chatterbox"] = ChatterboxTTS.from_pretrained(device=DEVICE)
    return _engines["chatterbox"]

# Preload Kokoro in background (fastest, most used)
threading.Thread(target=_load_kokoro, daemon=True).start()

# ── Kokoro voices ─────────────────────────────────────────────────────────────
KOKORO_VOICES = {
    # American English
    "af_heart":    {"label":"Heart",    "lang":"a","gender":"F","desc":"Feminina calorosa e natural"},
    "af_bella":    {"label":"Bella",    "lang":"a","gender":"F","desc":"Feminina suave e clara"},
    "af_sarah":    {"label":"Sarah",    "lang":"a","gender":"F","desc":"Feminina jovem e energética"},
    "af_nova":     {"label":"Nova",     "lang":"a","gender":"F","desc":"Feminina moderna e versátil"},
    "af_sky":      {"label":"Sky",      "lang":"a","gender":"F","desc":"Feminina suave e etérea"},
    "am_adam":     {"label":"Adam",     "lang":"a","gender":"M","desc":"Masculino profissional"},
    "am_michael":  {"label":"Michael",  "lang":"a","gender":"M","desc":"Masculino grave e autoritário"},
    "am_echo":     {"label":"Echo",     "lang":"a","gender":"M","desc":"Masculino neutro e claro"},
    "am_eric":     {"label":"Eric",     "lang":"a","gender":"M","desc":"Masculino jovem e dinâmico"},
    "am_liam":     {"label":"Liam",     "lang":"a","gender":"M","desc":"Masculino amigável"},
    # British English
    "bf_emma":     {"label":"Emma",     "lang":"b","gender":"F","desc":"Britânica elegante"},
    "bf_isabella": {"label":"Isabella", "lang":"b","gender":"F","desc":"Britânica sofisticada"},
    "bm_george":   {"label":"George",   "lang":"b","gender":"M","desc":"Britânico formal"},
    "bm_lewis":    {"label":"Lewis",    "lang":"b","gender":"M","desc":"Britânico natural"},
    # Portuguese BR
    "pf_dora":     {"label":"Dora",     "lang":"p","gender":"F","desc":"Portuguesa feminina"},
    "pm_alex":     {"label":"Alex BR",  "lang":"p","gender":"M","desc":"Português masculino BR"},
    "pm_santa":    {"label":"Santa",    "lang":"p","gender":"M","desc":"Português masculino grave"},
    # Spanish
    "ef_dora":     {"label":"Dora ES",  "lang":"e","gender":"F","desc":"Espanhola feminina"},
    "em_alex":     {"label":"Alex ES",  "lang":"e","gender":"M","desc":"Espanhol masculino"},
}

EDGE_VOICES = {
    "pt-BR-FranciscaNeural": {"label":"Francisca","lang":"pt","gender":"F","desc":"PT-BR feminina (premium)"},
    "pt-BR-AntonioNeural":   {"label":"Antonio",  "lang":"pt","gender":"M","desc":"PT-BR masculino (premium)"},
    "pt-BR-ThalitaMultilingualNeural":{"label":"Thalita","lang":"pt","gender":"F","desc":"PT-BR multilingual"},
    "pt-PT-RaquelNeural":    {"label":"Raquel",   "lang":"pt","gender":"F","desc":"PT-PT feminina"},
    "pt-PT-DuarteNeural":    {"label":"Duarte",   "lang":"pt","gender":"M","desc":"PT-PT masculino"},
    "en-US-JennyNeural":     {"label":"Jenny",    "lang":"en","gender":"F","desc":"EN feminina natural"},
    "en-US-GuyNeural":       {"label":"Guy",      "lang":"en","gender":"M","desc":"EN masculino profissional"},
    "en-US-AriaNeural":      {"label":"Aria",     "lang":"en","gender":"F","desc":"EN feminina expressiva"},
    "en-US-DavisNeural":     {"label":"Davis",    "lang":"en","gender":"M","desc":"EN masculino casual"},
    "en-GB-SoniaNeural":     {"label":"Sonia",    "lang":"en","gender":"F","desc":"British feminina"},
    "en-GB-RyanNeural":      {"label":"Ryan GB",  "lang":"en","gender":"M","desc":"British masculino"},
    "es-ES-ElviraNeural":    {"label":"Elvira",   "lang":"es","gender":"F","desc":"Espanhola premium"},
    "fr-FR-DeniseNeural":    {"label":"Denise",   "lang":"fr","gender":"F","desc":"Francesa natural"},
    "de-DE-KatjaNeural":     {"label":"Katja",    "lang":"de","gender":"F","desc":"Alemã natural"},
    "ja-JP-NanamiNeural":    {"label":"Nanami",   "lang":"ja","gender":"F","desc":"Japonesa premium"},
    "zh-CN-XiaoxiaoNeural":  {"label":"Xiaoxiao", "lang":"zh","gender":"F","desc":"Chinesa expressiva"},
}

EDGE_RATES  = ["-30%","-20%","-10%","+0%","+10%","+20%","+30%","+50%"]
EDGE_PITCHES= ["-10Hz","-5Hz","+0Hz","+5Hz","+10Hz","+15Hz"]

EQ_PRESETS = {
    "Neutro":   None,
    "Cinema":   {"bass":4,"mid":1,"treble":2},
    "Podcast":  {"bass":2,"mid":3,"treble":1},
    "Rádio":    {"bass":3,"mid":4,"treble":3},
    "Quente":   {"bass":4,"mid":0,"treble":-2},
    "Brilhante":{"bass":-2,"mid":1,"treble":5},
    "Telefone": {"bass":-8,"mid":5,"treble":-5},
    "Estúdio":  {"bass":1,"mid":2,"treble":3},
}

# ── State ─────────────────────────────────────────────────────────────────────
_history      = json.loads(HIST_FILE.read_text(encoding="utf-8")) if HIST_FILE.exists() else []
_pron         = json.loads(PRON_FILE.read_text(encoding="utf-8")) if PRON_FILE.exists() else {}
_saved_voices = json.loads(VOICES_FILE.read_text(encoding="utf-8")) if VOICES_FILE.exists() else {}
_stats        = defaultdict(float)
_api_keys     = {}

def _save_history():
    HIST_FILE.write_text(json.dumps(_history[-500:], ensure_ascii=False, indent=2), encoding="utf-8")
def _save_voices():
    VOICES_FILE.write_text(json.dumps(_saved_voices, ensure_ascii=False, indent=2), encoding="utf-8")
def _save_pron():
    PRON_FILE.write_text(json.dumps(_pron, ensure_ascii=False, indent=2), encoding="utf-8")

# ── Audio utils ───────────────────────────────────────────────────────────────
def _wav_to_b64(w: np.ndarray, sr: int) -> str:
    if w.ndim > 1: w = w.mean(-1)
    buf = io.BytesIO()
    sf.write(buf, w.astype(np.float32), sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode()

def _save_wav(w: np.ndarray, sr: int, prefix: str) -> Path:
    if w.ndim > 1: w = w.mean(-1)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = uuid.uuid4().hex[:6]
    path = OUT_DIR / f"{prefix}_{ts}_{uid}.wav"
    sf.write(str(path), w.astype(np.float32), sr)
    return path

def _add_history(entry: dict):
    entry.setdefault("id", uuid.uuid4().hex[:8])
    entry.setdefault("created", datetime.now().isoformat())
    _history.insert(0, entry)
    _stats["generated"] += 1
    _stats["chars"]     += len(entry.get("text", ""))
    _stats["seconds"]   += float(entry.get("duration_s", 0))
    _save_history()

def _audio_stats(w: np.ndarray, sr: int) -> dict:
    peak = float(np.max(np.abs(w))) if len(w) else 0.0
    rms  = float(np.sqrt(np.mean(w**2))) if len(w) else 0.0
    return {
        "duration_s":  round(float(len(w)) / sr, 2),
        "peak_db":     round(20 * np.log10(max(peak, 1e-9)), 1),
        "rms_db":      round(20 * np.log10(max(rms,  1e-9)), 1),
        "sample_rate": int(sr),
        "samples":     int(len(w)),
    }

def _waveform(w: np.ndarray, pts: int = 200) -> list:
    if len(w) == 0: return [0.0] * pts
    step = max(1, len(w) // pts)
    return [round(float(np.max(np.abs(w[i:i+step]))), 4) for i in range(0, len(w), step)][:pts]

# ── 50 Audio Processing Features ─────────────────────────────────────────────

def fx_normalize(w, target_db=-3.0):
    peak = np.max(np.abs(w))
    if peak < 1e-9: return w
    target_lin = 10 ** (target_db / 20)
    return w / peak * target_lin

def fx_speed(w, sr, rate):
    if not LIBROSA_OK or abs(rate - 1.0) < 0.01: return w, sr
    return librosa.effects.time_stretch(w.astype(np.float32), rate=rate), sr

def fx_pitch(w, sr, semitones):
    if not LIBROSA_OK or abs(semitones) < 0.1: return w
    return librosa.effects.pitch_shift(w.astype(np.float32), sr=sr, n_steps=semitones)

def fx_trim(w, sr, top_db=35):
    if not LIBROSA_OK: return w
    t, _ = librosa.effects.trim(w.astype(np.float32), top_db=top_db)
    return t

def fx_reverb(w, sr, amount=0.3):
    from scipy.signal import fftconvolve
    ir_len = int(sr * 0.6)
    t  = np.linspace(0, 0.6, ir_len)
    ir = np.exp(-6 * t) * np.random.randn(ir_len) * 0.5
    ir[0] = 1.0
    wet = fftconvolve(w.astype(np.float32), ir)[:len(w)]
    return (w * (1 - amount) + wet * amount).astype(np.float32)

def fx_echo(w, sr, delay_s=0.3, decay=0.4):
    d = int(sr * delay_s)
    out = np.zeros(len(w) + d, dtype=np.float32)
    out[:len(w)] = w
    out[d:] += w * decay
    return out[:len(w)]

def fx_denoise(w, sr):
    from scipy.signal import butter, sosfilt
    sos = butter(4, [80, 8000], btype="band", fs=sr, output="sos")
    return sosfilt(sos, w.astype(np.float32))

def fx_stereo(w, width=0.3):
    if w.ndim > 1: return w
    l = w + np.roll(w, 3) * width
    r = w - np.roll(w, 3) * width
    return np.stack([l, r], axis=-1).astype(np.float32)

def fx_eq(w, sr, bass=0, mid=0, treble=0):
    from scipy.signal import butter, sosfilt
    out = w.astype(np.float32)
    if abs(bass) > 0.1:
        sos = butter(2, 300, btype="low", fs=sr, output="sos")
        out = out + sosfilt(sos, out) * (10**(bass/20) - 1)
    if abs(treble) > 0.1:
        sos = butter(2, 4000, btype="high", fs=sr, output="sos")
        out = out + sosfilt(sos, out) * (10**(treble/20) - 1)
    return np.clip(out, -1.0, 1.0)

def fx_compress(w, threshold=0.5, ratio=4.0):
    out = w.copy()
    mask = np.abs(out) > threshold
    out[mask] = np.sign(out[mask]) * (threshold + (np.abs(out[mask]) - threshold) / ratio)
    return out

def fx_fade(w, sr, fade_in=0.1, fade_out=0.2):
    w = w.copy()
    fi, fo = int(sr * fade_in), int(sr * fade_out)
    if fi > 0: w[:fi] *= np.linspace(0, 1, fi)
    if fo > 0: w[-fo:] *= np.linspace(1, 0, fo)
    return w

def fx_padding(w, sr, pad_s=0.3):
    p = np.zeros(int(sr * pad_s), dtype=np.float32)
    return np.concatenate([p, w, p])

def fx_export_mp3(w, sr, path):
    if not PYDUB_OK: return path.replace(".mp3", ".wav")
    buf = io.BytesIO()
    sf.write(buf, w.astype(np.float32), sr, format="WAV")
    buf.seek(0)
    AudioSegment.from_wav(buf).export(path, format="mp3", bitrate="192k")
    return path

def fx_export_flac(w, sr, prefix):
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUT_DIR / f"{prefix}_{ts}.flac"
    sf.write(str(path), w.astype(np.float32), sr, format="FLAC")
    return path

def fx_srt(text, duration_s):
    words = text.split(); wps = len(words) / max(duration_s, 0.1)
    lines, t = [], 0.0
    for i in range(0, len(words), 8):
        chunk = " ".join(words[i:i+8])
        t_end = t + len(words[i:i+8]) / max(wps, 0.1)
        def fmt(s): h,r=divmod(int(s),3600); m,sec=divmod(r,60); ms=int((s%1)*1000); return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"
        lines.append(f"{i//8+1}\n{fmt(t)} --> {fmt(t_end)}\n{chunk}\n"); t = t_end
    return "\n".join(lines)

def fx_clean_text(text):
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    return re.sub(r"`[^`]*`", "", text).strip()

def fx_apply_pron(text):
    for word, pron in _pron.items():
        text = re.sub(r"\b" + re.escape(word) + r"\b", pron, text, flags=re.IGNORECASE)
    return text

def fx_text_stats(text):
    words = len(text.split()); chars = len(text); est = words / 2.8
    return {"words": words, "chars": chars, "est_seconds": round(est, 1),
            "est_time": f"{int(est//60)}:{int(est%60):02d}"}

def fx_chunk(text, max_chars=300):
    sents = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sents:
        if len(cur) + len(s) < max_chars: cur += (" " if cur else "") + s
        else:
            if cur: chunks.append(cur)
            cur = s
    if cur: chunks.append(cur)
    return chunks

def fx_detect_lang(text):
    pt = len(re.findall(r"\b(de|da|do|para|com|que|uma|por|não|são|também|mas)\b", text, re.I))
    en = len(re.findall(r"\b(the|and|for|are|with|this|that|from|have|been)\b", text, re.I))
    scores = {"portuguese": pt, "english": en}
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "english"

def fx_aging(w, sr, delta=10):
    if not LIBROSA_OK: return w
    w = librosa.effects.pitch_shift(w.astype(np.float32), sr=sr, n_steps=-delta*0.05)
    w, _ = fx_speed(w, sr, max(0.7, 1.0 - delta*0.003))
    return w

def fx_gender(w, sr, direction="female"):
    if not LIBROSA_OK: return w
    return librosa.effects.pitch_shift(w.astype(np.float32), sr=sr, n_steps=4 if direction=="female" else -4)

def fx_background(w, sr, music_path, vol=0.12):
    if not LIBROSA_OK: return w
    music, _ = librosa.load(music_path, sr=sr, mono=True)
    if len(music) < len(w): music = np.tile(music, int(np.ceil(len(w)/len(music))))
    return np.clip(w + music[:len(w)] * vol, -1.0, 1.0)

def fx_waveform_data(w, pts=200):
    if len(w) == 0: return [0.0]*pts
    step = max(1, len(w)//pts)
    return [round(float(np.max(np.abs(w[i:i+step]))), 4) for i in range(0, len(w), step)][:pts]

def fx_fingerprint(w):
    snippet = w[:1024] if len(w) >= 1024 else w
    return hashlib.md5(snippet.tobytes()).hexdigest()[:12]

def fx_cost_vs_elevenlabs(chars):
    eleven = round(chars / 1000 * 0.30, 4)
    return {"chars": chars, "elevenlabs_usd": eleven, "ours_usd": 0.0,
            "saved_usd": eleven, "saved_pct": 100}

def fx_pipeline(w, sr, opts):
    if opts.get("normalize"):       w = fx_normalize(w)
    if opts.get("trim"):            w = fx_trim(w, sr)
    speed = float(opts.get("speed", 1.0))
    if abs(speed - 1.0) > 0.01:    w, sr = fx_speed(w, sr, speed)
    pitch = float(opts.get("pitch", 0.0))
    if abs(pitch) > 0.1:           w = fx_pitch(w, sr, pitch)
    if opts.get("reverb"):          w = fx_reverb(w, sr, float(opts.get("reverb_amount", 0.3)))
    if opts.get("echo"):            w = fx_echo(w, sr)
    if opts.get("denoise"):         w = fx_denoise(w, sr)
    if opts.get("compress"):        w = fx_compress(w)
    eq = opts.get("eq", "Neutro")
    if eq and eq != "Neutro":
        p = EQ_PRESETS.get(eq) or {}
        if p: w = fx_eq(w, sr, p.get("bass",0), p.get("mid",0), p.get("treble",0))
    if opts.get("fade"):            w = fx_fade(w, sr)
    if opts.get("padding"):         w = fx_padding(w, sr)
    return fx_normalize(w), sr

# ── Engine: Kokoro ────────────────────────────────────────────────────────────
def gen_kokoro(text: str, voice: str, speed: float = 1.0, opts: dict = None) -> tuple:
    opts    = opts or {}
    kokoro  = _load_kokoro()
    vinfo   = KOKORO_VOICES.get(voice, {"lang": "a"})
    lang    = vinfo["lang"]
    pipe    = kokoro.get(lang, kokoro["a"])
    # apply pronunciation dict
    text = fx_apply_pron(fx_clean_text(text))
    segments = []
    for chunk in fx_chunk(text, 250):
        gen = pipe(chunk, voice=voice, speed=speed)
        for _, _, audio in gen:
            if audio is not None:
                segments.append(np.array(audio, dtype=np.float32))
    if not segments:
        raise RuntimeError("Kokoro não gerou áudio")
    sr = 24000
    silence = np.zeros(int(sr * 0.25), dtype=np.float32)
    w = np.concatenate([np.concatenate([s, silence]) for s in segments])
    w, sr = fx_pipeline(w, sr, opts)
    return w, sr

# ── Engine: F5-TTS (clonagem zero-shot) ───────────────────────────────────────
def gen_f5(text: str, ref_audio: str, ref_text: str = "", opts: dict = None) -> tuple:
    opts  = opts or {}
    model = _load_f5()
    text  = fx_apply_pron(fx_clean_text(text))
    wav, sr, _ = model.infer(ref_file=ref_audio, ref_text=ref_text, gen_text=text)
    w = np.array(wav, dtype=np.float32)
    w, sr = fx_pipeline(w, sr if isinstance(sr, int) else 24000, opts)
    return w, sr if isinstance(sr, int) else 24000

# ── Engine: Chatterbox ────────────────────────────────────────────────────────
def gen_chatterbox(text: str, ref_audio: str = None, exaggeration: float = 0.5, opts: dict = None) -> tuple:
    opts  = opts or {}
    model = _load_chatterbox()
    text  = fx_apply_pron(fx_clean_text(text))
    sr    = 24000
    if ref_audio:
        wav = model.generate(text, audio_prompt_path=ref_audio, exaggeration=exaggeration)
    else:
        wav = model.generate(text, exaggeration=exaggeration)
    w = np.array(wav.squeeze().cpu(), dtype=np.float32)
    w, sr = fx_pipeline(w, model.sr if hasattr(model, "sr") else sr, opts)
    return w, sr

# ── Engine: Edge TTS ─────────────────────────────────────────────────────────
def gen_edge(text: str, voice: str, rate: str = "+0%", pitch: str = "+0Hz", opts: dict = None) -> tuple:
    import edge_tts
    opts = opts or {}
    text = fx_apply_pron(fx_clean_text(text))
    tmp  = tempfile.mktemp(suffix=".mp3")
    def _thread():
        async def _run():
            comm = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
            await comm.save(tmp)
        asyncio.run(_run())
    t = threading.Thread(target=_thread)
    t.start(); t.join()
    w, sr = sf.read(tmp)
    os.unlink(tmp)
    w = w.astype(np.float32)
    if w.ndim > 1: w = w.mean(-1)
    w, sr = fx_pipeline(w, sr, opts)
    return w, int(sr)

# ── Master generate ───────────────────────────────────────────────────────────
def generate(engine: str, text: str, params: dict, opts: dict) -> dict:
    t0 = time.time()
    engine = engine.lower()

    if engine == "kokoro":
        voice = params.get("voice", "af_heart")
        speed = float(params.get("speed", 1.0))
        w, sr = gen_kokoro(text, voice, speed, opts)
        used_voice = voice

    elif engine == "f5":
        ref   = params.get("ref_audio", "")
        rtext = params.get("ref_text", "")
        if not ref or not Path(ref).exists():
            raise ValueError("F5-TTS precisa de um áudio de referência")
        w, sr = gen_f5(text, ref, rtext, opts)
        used_voice = "f5-clone"

    elif engine == "chatterbox":
        ref  = params.get("ref_audio")
        exag = float(params.get("exaggeration", 0.5))
        w, sr = gen_chatterbox(text, ref, exag, opts)
        used_voice = "chatterbox"

    elif engine == "edge":
        voice = params.get("voice", "pt-BR-FranciscaNeural")
        rate  = params.get("rate", "+0%")
        pitch = params.get("pitch", "+0Hz")
        w, sr = gen_edge(text, voice, rate, pitch, opts)
        used_voice = voice

    else:
        raise ValueError(f"Engine desconhecido: {engine}")

    # Save
    path = _save_wav(w, sr, f"{engine}_{used_voice.split('-')[0].lower()}")

    # Exports
    exports = {"wav": str(path)}
    if opts.get("export_mp3"):
        mp3 = str(path).replace(".wav", ".mp3")
        fx_export_mp3(w, sr, mp3); exports["mp3"] = mp3
    if opts.get("export_flac"):
        exports["flac"] = str(fx_export_flac(w, sr, engine))

    srt = None
    if opts.get("generate_srt"):
        astats = _audio_stats(w, sr)
        srt = fx_srt(text, astats["duration_s"])
        srt_path = str(path).replace(".wav", ".srt")
        Path(srt_path).write_text(srt, encoding="utf-8")
        exports["srt"] = srt_path

    duration_gen = round(time.time() - t0, 1)
    astats = _audio_stats(w, sr)

    _add_history({
        "text": text[:200], "engine": engine, "voice": used_voice,
        "path": str(path), "duration_s": duration_gen,
        "audio_s": astats["duration_s"],
    })

    return JSONResponse({
        "audio_b64":   _wav_to_b64(w, sr),
        "filename":    path.name,
        "path":        str(path),
        "engine":      engine,
        "voice":       used_voice,
        "duration_s":  duration_gen,
        "stats":       astats,
        "waveform":    fx_waveform_data(w),
        "exports":     exports,
        "srt":         srt,
        "fingerprint": fx_fingerprint(w),
        "cost":        fx_cost_vs_elevenlabs(len(text)),
    })

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="QWN3-TTS Studio", version="5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

async def _body(request: Request) -> dict:
    raw = await request.body()
    return json.loads(raw.decode("utf-8", errors="replace"))

@app.get("/", response_class=HTMLResponse)
async def root(): return _html()

@app.get("/api/info")
async def info():
    return {"gpu": GPU_NAME, "vram": GPU_VRAM, "version": "5.0",
            "engines_loaded": list(_engines.keys()),
            "engines_available": ["kokoro","f5","chatterbox","edge"],
            "saved_voices": len(_saved_voices)}

@app.post("/api/generate")
async def api_generate(request: Request):
    try:
        req    = await _body(request)
        engine = req.get("engine", "kokoro")
        text   = req.get("text", req.get("texto", ""))
        params = req.get("params", {})
        opts   = req.get("opts", {})
        if not text.strip():
            return JSONResponse({"error": "Texto vazio"}, status_code=400)
        return generate(engine, text, params, opts)
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/generate/clone")
async def api_clone(text: str = Form(...), engine: str = Form("f5"),
                    ref_text: str = Form(""), exaggeration: float = Form(0.5),
                    audio_ref: UploadFile = File(...)):
    tmp = None
    try:
        data = await audio_ref.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(data); tmp = f.name
        params = {"ref_audio": tmp, "ref_text": ref_text, "exaggeration": exaggeration}
        return generate(engine, text, params, {})
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if tmp and os.path.exists(tmp): os.unlink(tmp)

@app.post("/api/batch")
async def api_batch(request: Request):
    req    = await _body(request)
    texts  = req.get("texts", [])
    engine = req.get("engine", "kokoro")
    params = req.get("params", {})
    opts   = req.get("opts", {})
    results = []
    for i, text in enumerate(texts):
        if not text.strip(): continue
        try:
            r = generate(engine, text, params, opts)
            body = json.loads(r.body)
            results.append({"text": text[:60], "filename": body["filename"],
                            "audio_s": body["stats"]["duration_s"], "ok": True})
        except Exception as e:
            results.append({"text": text[:60], "error": str(e), "ok": False})
    return {"results": results, "total": len(results)}

@app.post("/api/podcast")
async def api_podcast(request: Request):
    req    = await _body(request)
    script = req.get("script", [])
    try:
        segments, sr = [], 24000
        for seg in script:
            if not seg.get("text","").strip(): continue
            engine = seg.get("engine","kokoro")
            params = {"voice": seg.get("voice","af_heart")}
            r    = generate(engine, seg["text"], params, {})
            body = json.loads(r.body)
            # decode base64 back to numpy
            wav_bytes = base64.b64decode(body["audio_b64"])
            w, _sr = sf.read(io.BytesIO(wav_bytes))
            sr = _sr
            segments.append(w.astype(np.float32))
        if not segments:
            return JSONResponse({"error":"Script vazio"}, status_code=400)
        silence = np.zeros(int(sr * 0.4), dtype=np.float32)
        combined = np.concatenate([np.concatenate([s, silence]) for s in segments])
        path = _save_wav(combined, sr, "podcast")
        return JSONResponse({"audio_b64": _wav_to_b64(combined, sr), "filename": path.name,
                             "duration_s": round(float(len(combined))/sr, 1)})
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/compare")
async def api_compare(request: Request):
    req = await _body(request)
    text = req.get("text", "")
    configs = req.get("configs", [])  # [{engine, params}]
    results = {}
    for cfg in configs:
        key = f"{cfg['engine']}:{cfg.get('params',{}).get('voice','')}"
        try:
            r = generate(cfg["engine"], text, cfg.get("params",{}), {})
            body = json.loads(r.body)
            results[key] = {"audio_b64": body["audio_b64"], "engine": cfg["engine"],
                            "stats": body["stats"]}
        except Exception as e:
            results[key] = {"error": str(e)}
    return results

# ── Voice Bank ────────────────────────────────────────────────────────────────
@app.get("/api/voices/saved")
async def list_saved(): return list(_saved_voices.values())

@app.post("/api/voices/saved")
async def save_voice(request: Request):
    req = await _body(request)
    vid = uuid.uuid4().hex[:8]
    entry = {"id": vid, "name": req.get("name", f"Voz {vid}"),
             "engine": req.get("engine","kokoro"), "params": req.get("params",{}),
             "description": req.get("description",""), "sample_path": req.get("sample_path",""),
             "created": datetime.now().isoformat(), "used": 0}
    _saved_voices[vid] = entry; _save_voices()
    return entry

@app.delete("/api/voices/saved/{vid}")
async def delete_voice(vid: str):
    _saved_voices.pop(vid, None); _save_voices(); return {"ok": True}

@app.post("/api/voices/saved/{vid}/generate")
async def generate_saved(vid: str, request: Request):
    req   = await _body(request)
    voice = _saved_voices.get(vid)
    if not voice: return JSONResponse({"error":"Não encontrada"}, status_code=404)
    voice["used"] += 1; _save_voices()
    return generate(voice["engine"], req.get("text",""), voice["params"], req.get("opts",{}))

# ── History ───────────────────────────────────────────────────────────────────
@app.get("/api/history")
async def history(q: str = ""):
    if q:
        return [h for h in _history if q.lower() in h.get("text","").lower()]
    return _history[:100]

@app.delete("/api/history/{eid}")
async def del_history(eid: str):
    global _history
    _history = [h for h in _history if h.get("id") != eid]
    _save_history(); return {"ok": True}

@app.get("/api/history/export")
async def export_history():
    path = OUT_DIR / f"history_{datetime.now().strftime('%Y%m%d')}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id","text","engine","voice","audio_s","duration_s","created","path"])
        w.writeheader()
        for h in _history: w.writerow({k: h.get(k,"") for k in w.fieldnames})
    return FileResponse(str(path), filename=path.name)

# ── Stats ─────────────────────────────────────────────────────────────────────
@app.get("/api/stats")
async def stats():
    return {"total_generated": int(_stats["generated"]), "total_chars": int(_stats["chars"]),
            "total_seconds": round(float(_stats["seconds"]),1), "history_count": len(_history),
            "projects": len(list(PROJ_DIR.glob("*.json"))),
            "saved_voices": len(_saved_voices), "engines": list(_engines.keys())}

# ── Projects ──────────────────────────────────────────────────────────────────
@app.get("/api/projects")
async def list_projects(): return [p.stem for p in PROJ_DIR.glob("*.json")]

@app.post("/api/projects")
async def save_project(request: Request):
    req  = await _body(request)
    name = req["name"].replace(" ","_")
    data = req.get("data", {}); data["saved_at"] = datetime.now().isoformat()
    (PROJ_DIR / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "name": name}

@app.get("/api/projects/{name}")
async def load_project(name: str):
    p = PROJ_DIR / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}

@app.delete("/api/projects/{name}")
async def delete_project(name: str):
    p = PROJ_DIR / f"{name}.json"
    if p.exists(): p.unlink()
    return {"ok": True}

# ── Pronunciations ────────────────────────────────────────────────────────────
@app.get("/api/pronunciations")
async def pronunciations(): return _pron

@app.post("/api/pronunciations")
async def add_pron(request: Request):
    req = await _body(request)
    _pron[req["word"]] = req["phonetic"]; _save_pron(); return {"ok": True}

@app.delete("/api/pronunciations/{word}")
async def del_pron(word: str):
    _pron.pop(word, None); _save_pron(); return {"ok": True}

# ── Audio serve ───────────────────────────────────────────────────────────────
@app.get("/api/audio/{filename}")
async def serve(filename: str):
    p = OUT_DIR / filename
    return FileResponse(str(p)) if p.exists() else JSONResponse({"error":"Not found"},status_code=404)

@app.post("/api/text/stats")
async def text_stats(request: Request):
    req = await _body(request); return fx_text_stats(req.get("text",""))

@app.get("/api/cost")
async def cost(chars: int = 1000): return fx_cost_vs_elevenlabs(chars)

# ── HTML ──────────────────────────────────────────────────────────────────────
def _html():
    kv = json.dumps(KOKORO_VOICES, ensure_ascii=False)
    ev = json.dumps(EDGE_VOICES,   ensure_ascii=False)
    eq = json.dumps(list(EQ_PRESETS.keys()), ensure_ascii=False)
    er = json.dumps(EDGE_RATES, ensure_ascii=False)
    ep = json.dumps(EDGE_PITCHES, ensure_ascii=False)
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>QWN3-TTS Studio v5</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
:root{{
  --bg:#060612;--s1:#0d0d1e;--s2:#131325;--s3:#1a1a2e;--bd:#252545;
  --t:#eaeaf5;--dim:#5a5a80;--a:#6d28d9;--a2:#9333ea;--a3:#c084fc;
  --g:#10b981;--r:#ef4444;--y:#f59e0b;--b:#3b82f6;
  --kokoro:#7c3aed;--f5:#0891b2;--chat:#059669;--edge:#d97706;
}}
body{{background:var(--bg);color:var(--t);font-family:'Inter',system-ui,sans-serif;height:100vh;display:flex;flex-direction:column;overflow:hidden}}
header{{background:linear-gradient(135deg,#1a0533,#0d0d1e);border-bottom:1px solid var(--bd);padding:12px 24px;display:flex;align-items:center;gap:14px;flex-shrink:0}}
.logo{{font-size:20px;font-weight:900;background:linear-gradient(135deg,#c084fc,#60a5fa,#34d399);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.tagline{{font-size:10px;color:var(--dim)}}
.badges{{display:flex;gap:6px;margin-left:auto;align-items:center}}
.badge{{background:var(--s2);border:1px solid var(--bd);padding:3px 10px;border-radius:20px;font-size:10px;color:var(--dim)}}
.badge.live{{color:var(--g);border-color:#10b98140}}
.app{{display:flex;flex:1;overflow:hidden}}
/* SIDEBAR */
.sidebar{{width:200px;background:var(--s1);border-right:1px solid var(--bd);display:flex;flex-direction:column;overflow-y:auto;flex-shrink:0}}
.ns{{padding:12px 12px 6px;font-size:9px;text-transform:uppercase;letter-spacing:1.5px;color:var(--dim)}}
.nb{{display:flex;align-items:center;gap:8px;padding:8px 14px;cursor:pointer;border:none;background:none;color:var(--dim);font-size:12px;width:100%;text-align:left;border-left:2px solid transparent;transition:all .15s}}
.nb:hover{{background:var(--s2);color:var(--t)}}
.nb.active{{background:var(--s2);color:var(--a3);border-left-color:var(--a)}}
/* MAIN */
.main{{flex:1;overflow-y:auto;padding:20px}}
.page{{display:none;max-width:1000px;margin:0 auto;animation:fi .2s ease}}
.page.active{{display:block}}
@keyframes fi{{from{{opacity:0;transform:translateY(5px)}}to{{opacity:1}}}}
.pt{{font-size:18px;font-weight:700;margin-bottom:3px}}
.ps{{font-size:12px;color:var(--dim);margin-bottom:16px}}
.g2{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
.g3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}}
@media(max-width:768px){{.g2,.g3{{grid-template-columns:1fr}}}}
.card{{background:var(--s1);border:1px solid var(--bd);border-radius:12px;padding:18px}}
.ct{{font-size:10px;font-weight:600;text-transform:uppercase;letter-spacing:1.2px;color:var(--dim);margin-bottom:12px}}
label{{font-size:11px;color:var(--dim);display:block;margin-bottom:4px;margin-top:10px}}
label:first-child{{margin-top:0}}
select,textarea,input[type=text],input[type=number],input[type=range]{{width:100%;background:var(--s2);border:1px solid var(--bd);color:var(--t);padding:8px 12px;border-radius:8px;font-size:12px;font-family:inherit;outline:none;transition:border .15s;-webkit-appearance:none}}
select:focus,textarea:focus,input:focus{{border-color:var(--a)}}
textarea{{resize:vertical;min-height:90px;line-height:1.6}}
input[type=range]{{padding:3px 0;cursor:pointer;accent-color:var(--a)}}
.rrow{{display:flex;align-items:center;gap:8px}}
.rv{{font-size:11px;color:var(--a3);min-width:40px;text-align:right}}
/* ENGINE TABS */
.etabs{{display:flex;gap:4px;margin-bottom:14px}}
.etab{{padding:7px 14px;border-radius:8px;cursor:pointer;border:1px solid var(--bd);background:var(--s2);color:var(--dim);font-size:12px;font-weight:600;font-family:inherit;transition:all .15s}}
.etab:hover{{color:var(--t)}}
.etab.active.kokoro{{background:#7c3aed20;border-color:var(--kokoro);color:#c084fc}}
.etab.active.f5{{background:#0891b220;border-color:var(--f5);color:#67e8f9}}
.etab.active.chat{{background:#05996920;border-color:var(--chat);color:#34d399}}
.etab.active.edge{{background:#d9770620;border-color:var(--edge);color:#fbbf24}}
.epanel{{display:none}}.epanel.active{{display:block}}
/* VOICE GRID */
.vgrid{{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin-bottom:8px}}
.vbtn{{background:var(--s2);border:2px solid var(--bd);color:var(--dim);padding:8px 5px;border-radius:9px;cursor:pointer;font-size:10px;text-align:center;transition:all .15s;width:100%;font-family:inherit}}
.vbtn:hover{{border-color:var(--a);color:var(--t)}}
.vbtn.sel{{border-color:var(--a);background:#6d28d918;color:var(--a3)}}
.vbtn .vn{{font-weight:700;display:block;font-size:11px}}
.vbtn .vd{{display:block;font-size:9px;margin-top:2px;color:var(--dim)}}
.vbtn.sel .vd{{color:var(--a3)}}
/* BUTTONS */
.btn{{background:linear-gradient(135deg,var(--a),var(--a2));color:#fff;border:none;padding:11px 18px;border-radius:9px;font-size:13px;font-weight:700;cursor:pointer;width:100%;transition:all .2s;margin-top:12px;font-family:inherit}}
.btn:hover{{transform:translateY(-1px);box-shadow:0 6px 20px #6d28d930}}
.btn:disabled{{opacity:.4;cursor:default;transform:none;box-shadow:none}}
.bsm{{background:var(--s2);border:1px solid var(--bd);color:var(--t);padding:6px 12px;border-radius:7px;cursor:pointer;font-size:11px;font-family:inherit;transition:all .15s}}
.bsm:hover{{border-color:var(--a);color:var(--a3)}}
.bsm.danger{{background:#ef444410;border-color:#ef444440;color:var(--r)}}
/* STATUS */
.status{{padding:9px 12px;border-radius:8px;font-size:12px;display:none;align-items:center;gap:8px;margin-top:8px}}
.status.show{{display:flex}}
.status.ok{{background:#10b98112;border:1px solid #10b98130;color:var(--g)}}
.status.err{{background:#ef444412;border:1px solid #ef444430;color:var(--r)}}
.status.loading{{background:#6d28d912;border:1px solid #6d28d930;color:var(--a3)}}
.spin{{width:14px;height:14px;border:2px solid #a855f730;border-top-color:var(--a3);border-radius:50%;animation:sp .7s linear infinite;flex-shrink:0}}
@keyframes sp{{to{{transform:rotate(360deg)}}}}
/* PLAYER */
.player{{background:var(--s2);border:1px solid var(--bd);border-radius:10px;padding:14px;margin-top:10px}}
audio{{width:100%;border-radius:6px;height:36px}}
canvas{{width:100%;height:52px;border-radius:7px;background:var(--s3);margin-top:8px}}
.ameta{{display:flex;gap:14px;margin-top:7px;font-size:10px;color:var(--dim)}}
.ameta span{{color:var(--t)}}
.arow{{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}}
/* TOGGLES */
.tgrid{{display:grid;grid-template-columns:1fr 1fr;gap:6px}}
.titem{{background:var(--s2);border:1px solid var(--bd);border-radius:8px;padding:9px 11px;cursor:pointer;display:flex;align-items:center;gap:7px;font-size:11px;transition:all .15s}}
.titem.on{{border-color:var(--a);background:#6d28d910;color:var(--a3)}}
.tdot{{width:8px;height:8px;border-radius:50%;background:var(--bd);flex-shrink:0;transition:background .15s}}
.titem.on .tdot{{background:var(--a3)}}
/* UPLOAD */
.upzone{{border:2px dashed var(--bd);border-radius:9px;padding:24px;text-align:center;cursor:pointer;position:relative;transition:all .2s}}
.upzone:hover,.upzone.drag{{border-color:var(--a);background:#6d28d908}}
.upzone.has{{border-color:var(--g);background:#10b98108}}
.upzone input{{position:absolute;inset:0;opacity:0;cursor:pointer}}
/* HISTORY */
.hitem{{background:var(--s2);border:1px solid var(--bd);border-radius:9px;padding:10px 12px;display:flex;align-items:center;gap:10px;margin-bottom:6px}}
.htxt{{flex:1;font-size:11px;color:var(--dim);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.hbadge{{background:var(--s3);padding:2px 7px;border-radius:10px;font-size:9px;color:var(--a3);flex-shrink:0}}
/* STAT CARD */
.sc{{background:var(--s2);border:1px solid var(--bd);border-radius:9px;padding:14px;text-align:center}}
.sn{{font-size:26px;font-weight:800;background:linear-gradient(135deg,var(--a2),var(--b));-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.sl{{font-size:10px;color:var(--dim);margin-top:3px}}
/* ENGINE DOT */
.edot{{display:inline-block;width:7px;height:7px;border-radius:50%;margin-right:5px}}
.edot.kokoro{{background:var(--kokoro)}}
.edot.f5{{background:var(--f5)}}
.edot.chat{{background:var(--chat)}}
.edot.edge{{background:var(--edge)}}
/* SCROLL */
::-webkit-scrollbar{{width:4px}}
::-webkit-scrollbar-thumb{{background:var(--bd);border-radius:3px}}
/* CHAR COUNT */
.ccount{{display:flex;justify-content:space-between;font-size:10px;color:var(--dim);margin-top:3px}}
</style>
</head>
<body>
<header>
  <div>
    <div class="logo">QWN3-TTS Studio</div>
    <div class="tagline">v5.0 · Kokoro · F5-TTS · Chatterbox · Edge TTS · Producao real</div>
  </div>
  <div class="badges">
    <div class="badge live" id="b-gpu">GPU: {GPU_NAME} {GPU_VRAM}</div>
    <div class="badge" id="b-engines">carregando...</div>
  </div>
</header>
<div class="app">
<nav class="sidebar">
  <div class="ns">Geração</div>
  <button class="nb active" onclick="nav('gen',this)">🎙️ Gerar</button>
  <button class="nb" onclick="nav('clone',this)">🔬 Clonar Voz</button>
  <button class="nb" onclick="nav('batch',this)">⚡ Batch</button>
  <button class="nb" onclick="nav('podcast',this)">🎙 Podcast</button>
  <button class="nb" onclick="nav('compare',this)">⚖️ Comparar</button>
  <div class="ns">Gerenciar</div>
  <button class="nb" onclick="nav('vbank',this)">🗄️ Banco de Vozes</button>
  <button class="nb" onclick="nav('history',this)">📋 Histórico</button>
  <button class="nb" onclick="nav('projects',this)">📁 Projetos</button>
  <button class="nb" onclick="nav('pron',this)">📖 Pronúncia</button>
  <button class="nb" onclick="nav('stats',this)">📊 Stats</button>
  <button class="nb" onclick="nav('api',this)">🔌 API</button>
</nav>
<div class="main">

<!-- GENERATE -->
<div id="page-gen" class="page active">
  <div class="pt">🎙️ Gerar Áudio</div>
  <div class="ps">4 engines de produção · Sem limite · Grátis</div>
  <div class="g2">
    <div>
      <!-- ENGINE SELECTOR -->
      <div class="card">
        <div class="ct">Engine</div>
        <div class="etabs">
          <button class="etab active kokoro" onclick="setEngine('kokoro',this)"><span class="edot kokoro"></span>Kokoro</button>
          <button class="etab f5"     onclick="setEngine('f5',this)"><span class="edot f5"></span>F5-TTS</button>
          <button class="etab chat"   onclick="setEngine('chatterbox',this)"><span class="edot chat"></span>Chatterbox</button>
          <button class="etab edge"   onclick="setEngine('edge',this)"><span class="edot edge"></span>Edge TTS</button>
        </div>
        <!-- Kokoro panel -->
        <div id="ep-kokoro" class="epanel active">
          <div class="ct" style="margin-top:4px">Voz Kokoro</div>
          <div class="vgrid" id="kvg"></div>
          <label>Velocidade</label>
          <div class="rrow"><input type="range" id="k-speed" min="0.5" max="2.0" step="0.05" value="1.0" oninput="rv('k-sv',this.value+'x')"><span class="rv" id="k-sv">1.0x</span></div>
        </div>
        <!-- F5 panel -->
        <div id="ep-f5" class="epanel">
          <div class="ct" style="margin-top:4px">F5-TTS — Clonagem Zero-Shot</div>
          <div style="font-size:11px;color:var(--dim);margin-bottom:10px">Faça upload de 5–30s do áudio que quer clonar</div>
          <div class="upzone" id="f5-uz"><input type="file" id="f5-ref" accept="audio/*" onchange="onUpload(this,'f5-uz','f5-txt')"><span style="font-size:24px">🎤</span><p id="f5-txt" style="font-size:12px;color:var(--dim);margin-top:6px">Áudio de referência (5–30s)</p></div>
          <label>O que o áudio diz (opcional mas melhora muito)</label>
          <textarea id="f5-rtext" rows="2" placeholder="Transcrição do áudio de referência..."></textarea>
        </div>
        <!-- Chatterbox panel -->
        <div id="ep-chatterbox" class="epanel">
          <div class="ct" style="margin-top:4px">Chatterbox — Controle Emocional</div>
          <div class="upzone" id="cb-uz"><input type="file" id="cb-ref" accept="audio/*" onchange="onUpload(this,'cb-uz','cb-txt')"><span style="font-size:24px">🎭</span><p id="cb-txt" style="font-size:12px;color:var(--dim);margin-top:6px">Referência de voz (opcional)</p></div>
          <label>Exageração emocional</label>
          <div class="rrow"><input type="range" id="cb-exag" min="0" max="1" step="0.05" value="0.5" oninput="rv('cb-ev',this.value)"><span class="rv" id="cb-ev">0.5</span></div>
        </div>
        <!-- Edge panel -->
        <div id="ep-edge" class="epanel">
          <div class="ct" style="margin-top:4px">Edge TTS — 322 Vozes Microsoft</div>
          <label>Voz</label>
          <select id="edge-voice"></select>
          <label>Velocidade</label>
          <select id="edge-rate"></select>
          <label>Tom</label>
          <select id="edge-pitch"></select>
        </div>
      </div>

      <!-- PROCESSING -->
      <div class="card" style="margin-top:14px">
        <div class="ct">Processamento</div>
        <label>Pitch (semitones)</label>
        <div class="rrow"><input type="range" id="g-pitch" min="-6" max="6" step="0.5" value="0" oninput="rv('g-pv',this.value)"><span class="rv" id="g-pv">0</span></div>
        <label>EQ Preset</label>
        <select id="g-eq"></select>
        <div style="margin-top:10px" class="tgrid">
          <div class="titem" onclick="tog(this,'normalize')"><div class="tdot"></div>Normalizar</div>
          <div class="titem" onclick="tog(this,'trim')"><div class="tdot"></div>Cortar Silêncio</div>
          <div class="titem" onclick="tog(this,'reverb')"><div class="tdot"></div>Reverb</div>
          <div class="titem" onclick="tog(this,'echo')"><div class="tdot"></div>Echo</div>
          <div class="titem" onclick="tog(this,'denoise')"><div class="tdot"></div>Denoise</div>
          <div class="titem" onclick="tog(this,'compress')"><div class="tdot"></div>Compressão</div>
          <div class="titem" onclick="tog(this,'fade')"><div class="tdot"></div>Fade In/Out</div>
          <div class="titem" onclick="tog(this,'padding')"><div class="tdot"></div>Padding</div>
        </div>
        <label style="margin-top:10px">Exportar</label>
        <div class="tgrid">
          <div class="titem" onclick="tog(this,'export_mp3')"><div class="tdot"></div>MP3</div>
          <div class="titem" onclick="tog(this,'generate_srt')"><div class="tdot"></div>Legendas SRT</div>
        </div>
      </div>
    </div>

    <div>
      <div class="card">
        <div class="ct">Texto</div>
        <textarea id="g-text" placeholder="Escreva aqui o texto para gerar áudio..." oninput="ccount(this,'g-cnt','g-words')"></textarea>
        <div class="ccount"><span id="g-words" style="color:var(--a3)"></span><span><span id="g-cnt">0</span> chars</span></div>
        <button class="btn" id="g-btn" onclick="doGenerate()">⚡ GERAR ÁUDIO</button>
      </div>
      <div id="g-status" class="status"></div>
      <div id="g-result" style="display:none">
        <div class="player">
          <audio id="g-audio" controls></audio>
          <canvas id="g-wave"></canvas>
          <div class="ameta">Engine: <span id="r-engine">-</span> &nbsp; Dur: <span id="r-dur">-</span> &nbsp; Peak: <span id="r-peak">-</span></div>
        </div>
        <div class="arow" style="margin-top:8px">
          <a id="g-dl" class="bsm" download>💾 WAV</a>
          <a id="g-dl-mp3" class="bsm" style="display:none" download>🎵 MP3</a>
          <a id="g-dl-srt" class="bsm" style="display:none" download>📄 SRT</a>
          <button class="bsm" onclick="saveToBank()">🗄️ Salvar Voz</button>
          <button class="bsm" onclick="showCost()">💰 Economia</button>
        </div>
        <div id="g-cost" style="display:none;margin-top:6px;padding:9px;background:var(--s2);border-radius:8px;font-size:11px;line-height:1.8"></div>
      </div>
    </div>
  </div>
</div>

<!-- CLONE -->
<div id="page-clone" class="page">
  <div class="pt">🔬 Clonar Voz</div>
  <div class="ps">F5-TTS ou Chatterbox — clonagem zero-shot com qualquer áudio</div>
  <div class="g2">
    <div class="card">
      <label>Engine de Clonagem</label>
      <select id="cl-engine"><option value="f5">F5-TTS (melhor fidelidade)</option><option value="chatterbox">Chatterbox (controle emocional)</option></select>
      <label>Áudio de referência</label>
      <div class="upzone" id="cl-uz"><input type="file" id="cl-ref" accept="audio/*" onchange="onUpload(this,'cl-uz','cl-txt')"><span style="font-size:28px">🎤</span><p id="cl-txt" style="font-size:12px;color:var(--dim);margin-top:8px">Qualquer áudio · Sem limite de tamanho</p></div>
      <label>Transcrição (melhora F5-TTS)</label>
      <textarea id="cl-rtext" rows="2" placeholder="O que o áudio de referência diz..."></textarea>
      <label>Texto novo</label>
      <textarea id="cl-text" placeholder="Este texto será falado com a voz clonada..."></textarea>
      <button class="btn" id="cl-btn" onclick="doClone()">🔬 CLONAR E GERAR</button>
    </div>
    <div>
      <div id="cl-status" class="status"></div>
      <div id="cl-result" style="display:none" class="player">
        <audio id="cl-audio" controls></audio>
        <div class="arow"><a id="cl-dl" class="bsm" download>💾 Baixar WAV</a></div>
      </div>
    </div>
  </div>
</div>

<!-- BATCH -->
<div id="page-batch" class="page">
  <div class="pt">⚡ Batch — Geração em Massa</div>
  <div class="ps">Gere centenas de áudios sem limites</div>
  <div class="g2">
    <div class="card">
      <label>Engine</label>
      <select id="bt-engine">
        <option value="kokoro">Kokoro</option><option value="edge">Edge TTS</option>
        <option value="chatterbox">Chatterbox</option>
      </select>
      <label>Voz Kokoro</label><select id="bt-voice"></select>
      <label>Textos (um por linha)</label>
      <textarea id="bt-texts" rows="12" placeholder="Primeira linha&#10;Segunda linha&#10;Terceira linha..."></textarea>
      <button class="btn" id="bt-btn" onclick="doBatch()">⚡ GERAR TUDO</button>
    </div>
    <div class="card"><div class="ct">Resultados</div><div id="bt-results"></div></div>
  </div>
</div>

<!-- PODCAST -->
<div id="page-podcast" class="page">
  <div class="pt">🎙 Podcast Multi-Speaker</div>
  <div class="ps">Crie conversas com vozes diferentes — filmes, séries, audiobooks</div>
  <div class="g2">
    <div class="card">
      <div class="ct">Script</div>
      <div id="pod-lines"></div>
      <button class="bsm" onclick="addPodLine()" style="margin-top:8px;width:100%">+ Adicionar fala</button>
      <button class="btn" onclick="doPodcast()" id="pod-btn">🎙 GERAR PODCAST</button>
    </div>
    <div>
      <div id="pod-status" class="status"></div>
      <div id="pod-result" style="display:none" class="player">
        <audio id="pod-audio" controls></audio>
        <div class="arow"><a id="pod-dl" class="bsm" download>💾 Baixar WAV</a></div>
      </div>
    </div>
  </div>
</div>

<!-- COMPARE -->
<div id="page-compare" class="page">
  <div class="pt">⚖️ Comparar Vozes</div>
  <div class="ps">Teste o mesmo texto com engines e vozes diferentes lado a lado</div>
  <div class="card" style="max-width:640px">
    <label>Texto</label>
    <textarea id="cmp-text" rows="3" placeholder="Mesmo texto gerado com configurações diferentes..."></textarea>
    <div class="g2" style="margin-top:10px">
      <div>
        <label>A — Engine</label><select id="cmp-a-engine"><option value="kokoro">Kokoro</option><option value="edge">Edge TTS</option></select>
        <label>A — Voz</label><select id="cmp-a-voice"></select>
      </div>
      <div>
        <label>B — Engine</label><select id="cmp-b-engine"><option value="kokoro">Kokoro</option><option value="edge">Edge TTS</option></select>
        <label>B — Voz</label><select id="cmp-b-voice"></select>
      </div>
    </div>
    <button class="btn" onclick="doCompare()">⚖️ COMPARAR</button>
  </div>
  <div id="cmp-results" class="g2" style="margin-top:14px"></div>
</div>

<!-- VOICE BANK -->
<div id="page-vbank" class="page">
  <div class="pt">🗄️ Banco de Vozes</div>
  <div class="ps">Salve configurações de engine+voz para reusar em qualquer projeto</div>
  <div class="g2">
    <div class="card">
      <div class="ct">Salvar Configuração Atual</div>
      <label>Nome</label><input type="text" id="vb-name" placeholder="Narrador Principal, Personagem A...">
      <label>Descrição</label><input type="text" id="vb-desc" placeholder="Para que serve...">
      <button class="btn" onclick="saveVoiceToBank()">🗄️ Salvar no Banco</button>
      <div id="vb-status" class="status" style="margin-top:8px"></div>
    </div>
    <div class="card">
      <div class="ct">Vozes Salvas <span id="vb-count" style="color:var(--a3)"></span></div>
      <div id="vb-list"><div style="color:var(--dim);font-size:12px">Nenhuma voz salva ainda.</div></div>
    </div>
  </div>
</div>

<!-- HISTORY -->
<div id="page-history" class="page">
  <div class="pt">📋 Histórico</div>
  <div class="ps">Todos os áudios gerados — nunca perde nada</div>
  <div style="display:flex;gap:8px;margin-bottom:14px">
    <input type="text" id="h-search" placeholder="Buscar..." oninput="loadHistory(this.value)" style="flex:1">
    <button class="bsm" onclick="exportHistory()">📥 CSV</button>
  </div>
  <div id="h-list"></div>
</div>

<!-- PROJECTS -->
<div id="page-projects" class="page">
  <div class="pt">📁 Projetos</div>
  <div class="ps">Salve todo o contexto de um projeto para não perder nada</div>
  <div class="g2">
    <div class="card">
      <div class="ct">Salvar Projeto</div>
      <label>Nome</label><input type="text" id="pj-name" placeholder="Filme X — Voz do Narrador">
      <button class="btn" onclick="saveProject()">💾 Salvar</button>
    </div>
    <div class="card"><div class="ct">Projetos Salvos</div><div id="pj-list"></div></div>
  </div>
</div>

<!-- PRONUNCIATION -->
<div id="page-pron" class="page">
  <div class="pt">📖 Dicionário de Pronúncia</div>
  <div class="ps">Ensine a IA a pronunciar nomes, marcas e termos específicos</div>
  <div class="g2">
    <div class="card">
      <label>Palavra original</label><input type="text" id="pr-word" placeholder="QWN3">
      <label>Pronunciar como</label><input type="text" id="pr-as" placeholder="Quê-Dáblio-En-Três">
      <button class="btn" onclick="addPron()">+ Adicionar</button>
    </div>
    <div class="card"><div class="ct">Dicionário</div><div id="pr-list"></div></div>
  </div>
</div>

<!-- STATS -->
<div id="page-stats" class="page">
  <div class="pt">📊 Estatísticas</div>
  <div class="ps">Uso e economia vs ElevenLabs</div>
  <div class="g3" id="st-cards" style="margin-bottom:14px"></div>
  <div class="card"><div class="ct">💰 Economia vs ElevenLabs</div><div id="st-cost"></div></div>
</div>

<!-- API -->
<div id="page-api" class="page">
  <div class="pt">🔌 API REST</div>
  <div class="ps">Integre o QWN3-TTS em qualquer pipeline de produção</div>
  <div class="card">
    <pre style="background:var(--s2);padding:14px;border-radius:9px;font-size:11px;color:var(--a3);overflow-x:auto">POST /api/generate          — Gerar (engine: kokoro|f5|chatterbox|edge)
POST /api/generate/clone    — Clonar com upload de áudio (multipart)
POST /api/batch             — Batch sem limites
POST /api/podcast           — Multi-speaker
POST /api/compare           — A/B engines
GET  /api/voices/saved      — Banco de vozes
POST /api/voices/saved      — Salvar voz
GET  /api/history           — Histórico (até 500 entradas)
GET  /api/stats             — Estatísticas
GET  /api/audio/{{file}}     — Servir áudio</pre>
    <label style="margin-top:14px">Exemplo (Kokoro)</label>
    <pre style="background:var(--s2);padding:12px;border-radius:8px;font-size:10px;overflow-x:auto">curl -X POST http://localhost:7860/api/generate \\
  -H "Content-Type: application/json" \\
  -d '{{"engine":"kokoro","text":"Olá!","params":{{"voice":"af_heart","speed":1.0}}}}'</pre>
    <label style="margin-top:14px">Exemplo (Edge TTS)</label>
    <pre style="background:var(--s2);padding:12px;border-radius:8px;font-size:10px;overflow-x:auto">curl -X POST http://localhost:7860/api/generate \\
  -H "Content-Type: application/json" \\
  -d '{{"engine":"edge","text":"Hello world","params":{{"voice":"en-US-AriaNeural","rate":"+0%"}}}}'</pre>
  </div>
</div>

</div>
</div>
<script>
const KV={kv};
const EV={ev};
const EQ={eq};
const ER={er};
const EP={ep};
let curEngine='kokoro', curKVoice='af_heart', curEdgeVoice='pt-BR-FranciscaNeural';
let toggles={{}}, lastResult=null, podLines=[];

// ── INIT ─────────────────────────────────────────────────────────────────────
window.onload=async()=>{{
  // Info
  try{{
    const i=await fetch('/api/info').then(r=>r.json());
    document.getElementById('b-engines').textContent=i.engines_loaded.length+' engines ativos';
    document.getElementById('b-engines').className='badge live';
  }}catch{{}}

  // Kokoro voice grid
  const vg=document.getElementById('kvg');
  Object.entries(KV).forEach(([k,v])=>{{
    const b=document.createElement('button');
    b.className='vbtn'+(k===curKVoice?' sel':'');
    b.innerHTML=`<span class="vn">${{v.label}}</span><span class="vd">${{v.desc.substring(0,20)}}</span>`;
    b.onclick=()=>{{document.querySelectorAll('#kvg .vbtn').forEach(x=>x.classList.remove('sel'));b.classList.add('sel');curKVoice=k;}};
    vg.appendChild(b);
  }});

  // Edge voice select
  const es=document.getElementById('edge-voice');
  Object.entries(EV).forEach(([k,v])=>{{
    const o=document.createElement('option');o.value=k;o.textContent=v.label+' ('+v.lang+') - '+v.desc;
    if(k===curEdgeVoice)o.selected=true; es.appendChild(o);
  }});
  es.onchange=()=>curEdgeVoice=es.value;

  // Edge rate/pitch
  const er=document.getElementById('edge-rate');
  ER.forEach(r=>{{const o=document.createElement('option');o.value=r;o.textContent=r;if(r==='+0%')o.selected=true;er.appendChild(o);}});
  const epitch=document.getElementById('edge-pitch');
  EP.forEach(p=>{{const o=document.createElement('option');o.value=p;o.textContent=p;if(p==='+0Hz')o.selected=true;epitch.appendChild(o);}});

  // EQ
  const eq=document.getElementById('g-eq');
  EQ.forEach(e=>{{const o=document.createElement('option');o.value=e;o.textContent=e;eq.appendChild(o);}});

  // Batch voice select
  const bv=document.getElementById('bt-voice');
  Object.entries(KV).forEach(([k,v])=>{{const o=document.createElement('option');o.value=k;o.textContent=v.label;bv.appendChild(o);}});

  // Compare voice selects
  ['cmp-a-voice','cmp-b-voice'].forEach(id=>{{
    const s=document.getElementById(id);
    Object.entries(KV).forEach(([k,v])=>{{const o=document.createElement('option');o.value=k;o.textContent=v.label;s.appendChild(o);}});
    if(id==='cmp-b-voice')s.value='am_adam';
  }});

  // Podcast defaults
  addPodLine(); addPodLine();

  loadHistory(); loadProjects(); loadPron(); loadStats(); loadVoiceBank();
}};

// ── NAV ──────────────────────────────────────────────────────────────────────
function nav(page,el){{
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  document.querySelectorAll('.nb').forEach(b=>b.classList.remove('active'));
  document.getElementById('page-'+page).classList.add('active');
  el.classList.add('active');
  if(page==='history')loadHistory();
  if(page==='stats')loadStats();
  if(page==='projects')loadProjects();
  if(page==='pron')loadPron();
  if(page==='vbank')loadVoiceBank();
}}

// ── ENGINE ───────────────────────────────────────────────────────────────────
function setEngine(e,btn){{
  curEngine=e;
  document.querySelectorAll('.etab').forEach(b=>b.classList.remove('active','kokoro','f5','chat','edge'));
  btn.classList.add('active',e==='chatterbox'?'chat':e);
  document.querySelectorAll('.epanel').forEach(p=>p.classList.remove('active'));
  document.getElementById('ep-'+e)?.classList.add('active');
  if(e==='edge')document.getElementById('ep-edge').classList.add('active');
}}

// ── HELPERS ──────────────────────────────────────────────────────────────────
function rv(id,v){{document.getElementById(id).textContent=v;}}
function tog(el,k){{el.classList.toggle('on');toggles[k]=el.classList.contains('on');}}
function setStatus(id,type,msg){{const e=document.getElementById(id);e.className='status show '+type;e.innerHTML=type==='loading'?`<div class="spin"></div>${{msg}}`:msg;}}
function b64Blob(b64){{const bin=atob(b64);const a=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)a[i]=bin.charCodeAt(i);return new Blob([a],{{type:'audio/wav'}});}}
function onUpload(inp,zoneId,txtId){{const f=inp.files?.[0];if(!f)return;document.getElementById(zoneId).classList.add('has');document.getElementById(txtId).textContent='✅ '+f.name;}}
function ccount(el,cid,wid){{const c=el.value.length;const w=el.value.trim().split(/\\s+/).filter(Boolean).length;document.getElementById(cid).textContent=c;document.getElementById(wid).textContent=w+' palavras';}}

function drawWave(data, canvasId){{
  const canvas=document.getElementById(canvasId);
  if(!canvas||!data)return;
  const ctx=canvas.getContext('2d');
  canvas.width=canvas.offsetWidth*2; canvas.height=canvas.offsetHeight*2;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const grad=ctx.createLinearGradient(0,0,canvas.width,0);
  grad.addColorStop(0,'#6d28d9');grad.addColorStop(0.5,'#9333ea');grad.addColorStop(1,'#60a5fa');
  ctx.fillStyle=grad;
  const bw=canvas.width/data.length, mid=canvas.height/2;
  data.forEach((v,i)=>{{const h=Math.max(2,v*mid*0.9);ctx.fillRect(i*bw,mid-h,bw*0.7,h*2);}});
}}

function showResult(d, audioId, dlId, statusId, resultId, waveId){{
  const url=URL.createObjectURL(b64Blob(d.audio_b64));
  document.getElementById(audioId).src=url;
  document.getElementById(dlId).href=url; document.getElementById(dlId).download=d.filename;
  document.getElementById(resultId).style.display='block';
  if(waveId)drawWave(d.waveform,waveId);
  setStatus(statusId,'ok',`✅ ${{d.engine||''}} · ${{d.stats?.duration_s||0}}s · ${{d.duration_s}}s geração`);
  lastResult=d;
  // MP3
  if(d.exports?.mp3){{const mp3=document.getElementById('g-dl-mp3');if(mp3){{mp3.href='/api/audio/'+d.exports.mp3.split(/[\\/\\\\]/).pop();mp3.style.display='inline-flex';}}}}
  if(d.exports?.srt){{const srt=document.getElementById('g-dl-srt');if(srt){{srt.href='/api/audio/'+d.exports.srt.split(/[\\/\\\\]/).pop();srt.style.display='inline-flex';}}}}
  // Meta
  const re=document.getElementById('r-engine');if(re)re.textContent=d.engine||'-';
  const rd=document.getElementById('r-dur');if(rd)rd.textContent=(d.stats?.duration_s||0)+'s';
  const rp=document.getElementById('r-peak');if(rp)rp.textContent=(d.stats?.peak_db||0)+'dB';
}}

// ── GENERATE ─────────────────────────────────────────────────────────────────
async function doGenerate(){{
  const text=document.getElementById('g-text').value.trim();
  if(!text)return;
  let params={{}};
  if(curEngine==='kokoro') params={{voice:curKVoice,speed:parseFloat(document.getElementById('k-speed').value)}};
  else if(curEngine==='f5'){{const f=document.getElementById('f5-ref').files?.[0];if(!f){{alert('Selecione um áudio de referência para F5-TTS');return;}} params={{ref_text:document.getElementById('f5-rtext').value}};}}
  else if(curEngine==='chatterbox') params={{exaggeration:parseFloat(document.getElementById('cb-exag').value)}};
  else if(curEngine==='edge') params={{voice:curEdgeVoice,rate:document.getElementById('edge-rate').value,pitch:document.getElementById('edge-pitch').value}};

  const pitch=parseFloat(document.getElementById('g-pitch').value);
  const opts={{...toggles,pitch,eq:document.getElementById('g-eq').value}};

  // F5 / Chatterbox need clone endpoint (ref file)
  if((curEngine==='f5'||curEngine==='chatterbox')&&document.getElementById(curEngine==='f5'?'f5-ref':'cb-ref').files?.[0]){{
    const fd=new FormData();
    fd.append('text',text);fd.append('engine',curEngine);
    if(curEngine==='f5'){{fd.append('ref_text',document.getElementById('f5-rtext').value||'');fd.append('audio_ref',document.getElementById('f5-ref').files[0]);}}
    else{{fd.append('exaggeration',''+parseFloat(document.getElementById('cb-exag').value));fd.append('audio_ref',document.getElementById('cb-ref').files[0]);}}
    document.getElementById('g-btn').disabled=true;
    setStatus('g-status','loading','⏳ Clonando voz...');
    try{{
      const d=await fetch('/api/generate/clone',{{method:'POST',body:fd}}).then(r=>r.json());
      if(d.error){{setStatus('g-status','err','❌ '+d.error);return;}}
      showResult(d,'g-audio','g-dl','g-status','g-result','g-wave');
    }}catch(e){{setStatus('g-status','err','❌ '+e.message);}}
    finally{{document.getElementById('g-btn').disabled=false;}}
    return;
  }}

  document.getElementById('g-btn').disabled=true;
  document.getElementById('g-result').style.display='none';
  setStatus('g-status','loading','⏳ Gerando com '+curEngine+'...');
  try{{
    const d=await fetch('/api/generate',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{engine:curEngine,text,params,opts}})}}).then(r=>r.json());
    if(d.error){{setStatus('g-status','err','❌ '+d.error);return;}}
    showResult(d,'g-audio','g-dl','g-status','g-result','g-wave');
  }}catch(e){{setStatus('g-status','err','❌ '+e.message);}}
  finally{{document.getElementById('g-btn').disabled=false;}}
}}

function showCost(){{
  if(!lastResult)return;
  const c=lastResult.cost;
  const el=document.getElementById('g-cost');
  el.style.display='block';
  el.innerHTML=`ElevenLabs cobraria: <b style="color:var(--r)">$${{c?.elevenlabs_usd||0}}</b> por ${{c?.chars||0}} chars<br>QWN3-TTS: <b style="color:var(--g)">$0.00</b> — economia de 100%`;
}}

// ── CLONE ────────────────────────────────────────────────────────────────────
async function doClone(){{
  const file=document.getElementById('cl-ref').files?.[0];
  if(!file){{alert('Selecione um áudio de referência');return;}}
  const text=document.getElementById('cl-text').value.trim();
  if(!text){{alert('Escreva o texto');return;}}
  const fd=new FormData();
  fd.append('text',text);fd.append('engine',document.getElementById('cl-engine').value);
  fd.append('ref_text',document.getElementById('cl-rtext').value||'');
  fd.append('audio_ref',file);
  document.getElementById('cl-btn').disabled=true;
  setStatus('cl-status','loading','🔬 Clonando voz...');
  try{{
    const d=await fetch('/api/generate/clone',{{method:'POST',body:fd}}).then(r=>r.json());
    if(d.error){{setStatus('cl-status','err','❌ '+d.error);return;}}
    const url=URL.createObjectURL(b64Blob(d.audio_b64));
    document.getElementById('cl-audio').src=url;
    document.getElementById('cl-dl').href=url;document.getElementById('cl-dl').download=d.filename;
    document.getElementById('cl-result').style.display='block';
    setStatus('cl-status','ok','✅ Voz clonada em '+d.duration_s+'s');
  }}catch(e){{setStatus('cl-status','err','❌ '+e.message);}}
  finally{{document.getElementById('cl-btn').disabled=false;}}
}}

// ── BATCH ────────────────────────────────────────────────────────────────────
async function doBatch(){{
  const texts=document.getElementById('bt-texts').value.split('\\n').filter(t=>t.trim());
  if(!texts.length)return;
  const engine=document.getElementById('bt-engine').value;
  const params={{voice:document.getElementById('bt-voice').value}};
  document.getElementById('bt-btn').disabled=true;
  document.getElementById('bt-results').innerHTML='<div style="color:var(--dim)">⏳ Gerando '+texts.length+' áudios...</div>';
  try{{
    const d=await fetch('/api/batch',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{engine,texts,params,opts:{{}}}})}}).then(r=>r.json());
    const div=document.getElementById('bt-results');div.innerHTML='';
    d.results.forEach(r=>{{
      const el=document.createElement('div');el.className='hitem';
      el.innerHTML=r.ok?`<span style="color:var(--g)">✓</span><span class="htxt">${{r.text}}</span><span style="font-size:10px;color:var(--dim)">${{r.audio_s}}s</span><a class="bsm" href="/api/audio/${{r.filename}}" download>💾</a>`:`<span style="color:var(--r)">✗</span><span class="htxt">${{r.text}} — ${{r.error}}</span>`;
      div.appendChild(el);
    }});
  }}catch(e){{document.getElementById('bt-results').innerHTML='<div style="color:var(--r)">'+e.message+'</div>';}}
  finally{{document.getElementById('bt-btn').disabled=false;}}
}}

// ── PODCAST ──────────────────────────────────────────────────────────────────
function addPodLine(){{
  const i=podLines.length;
  podLines.push({{engine:'kokoro',voice:'af_heart',text:''}});
  const div=document.getElementById('pod-lines');
  const row=document.createElement('div');
  row.className='hitem';row.id='pl-'+i;row.style.flexDirection='column';row.style.alignItems='stretch';row.style.gap='6px';
  row.innerHTML=`<div style="display:flex;gap:6px;align-items:center">
    <select onchange="podLines[${{i}}].engine=this.value" style="width:120px"><option value="kokoro">Kokoro</option><option value="edge">Edge</option></select>
    <select onchange="podLines[${{i}}].voice=this.value" style="flex:1">${{Object.entries(KV).map(([k,v])=>`<option value="${{k}}">${{v.label}}</option>`).join('')}}</select>
    <button class="bsm danger" onclick="document.getElementById('pl-${{i}}')?.remove();podLines[${{i}}]=null">✕</button>
  </div>
  <textarea rows="2" placeholder="Fala desta pessoa..." oninput="podLines[${{i}}].text=this.value" style="min-height:55px"></textarea>`;
  div.appendChild(row);
}}
async function doPodcast(){{
  const script=podLines.filter(Boolean).filter(l=>l.text?.trim());
  if(!script.length)return;
  document.getElementById('pod-btn').disabled=true;
  setStatus('pod-status','loading','🎙 Gerando podcast...');
  try{{
    const d=await fetch('/api/podcast',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{script}})}}).then(r=>r.json());
    if(d.error){{setStatus('pod-status','err','❌ '+d.error);return;}}
    const url=URL.createObjectURL(b64Blob(d.audio_b64));
    document.getElementById('pod-audio').src=url;document.getElementById('pod-dl').href=url;document.getElementById('pod-dl').download=d.filename;
    document.getElementById('pod-result').style.display='block';
    setStatus('pod-status','ok','✅ Podcast gerado! '+d.duration_s+'s');
  }}catch(e){{setStatus('pod-status','err','❌ '+e.message);}}
  finally{{document.getElementById('pod-btn').disabled=false;}}
}}

// ── COMPARE ──────────────────────────────────────────────────────────────────
async function doCompare(){{
  const text=document.getElementById('cmp-text').value.trim();if(!text)return;
  const configs=[
    {{engine:document.getElementById('cmp-a-engine').value,params:{{voice:document.getElementById('cmp-a-voice').value}}}},
    {{engine:document.getElementById('cmp-b-engine').value,params:{{voice:document.getElementById('cmp-b-voice').value}}}},
  ];
  const div=document.getElementById('cmp-results');div.innerHTML='<div style="color:var(--dim)">⏳ Gerando...</div>';
  try{{
    const d=await fetch('/api/compare',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{text,configs}})}}).then(r=>r.json());
    div.innerHTML='';
    Object.entries(d).forEach(([key,data])=>{{
      const c=document.createElement('div');c.className='player';
      if(data.error){{c.innerHTML=`<div style="color:var(--r)">${{key}}: ${{data.error}}</div>`;}}
      else{{const url=URL.createObjectURL(b64Blob(data.audio_b64));c.innerHTML=`<div style="font-weight:700;font-size:12px;margin-bottom:6px">${{key}}</div><audio src="${{url}}" controls></audio>`;}}
      div.appendChild(c);
    }});
  }}catch(e){{div.innerHTML='<div style="color:var(--r)">'+e.message+'</div>';}}
}}

// ── VOICE BANK ───────────────────────────────────────────────────────────────
async function loadVoiceBank(){{
  const data=await fetch('/api/voices/saved').then(r=>r.json()).catch(()=>[]);
  const div=document.getElementById('vb-list');
  const cnt=document.getElementById('vb-count');
  if(cnt)cnt.textContent=data.length?`(${{data.length}})`:'';
  if(!div)return;
  if(!data.length){{div.innerHTML='<div style="color:var(--dim);font-size:12px">Nenhuma voz salva ainda.</div>';return;}}
  div.innerHTML='';
  data.forEach(v=>{{
    const el=document.createElement('div');
    el.style.cssText='background:var(--s3);border:1px solid var(--bd);border-radius:9px;padding:10px;margin-bottom:8px';
    const edot=`<span class="edot ${{v.engine==='chatterbox'?'chat':v.engine}}"></span>`;
    el.innerHTML=`
      <div style="display:flex;align-items:center;gap:6px;margin-bottom:5px">
        ${{edot}}<span style="font-weight:700;font-size:13px">${{v.name}}</span>
        <span class="hbadge">${{v.engine}}</span>
        <span style="margin-left:auto;font-size:10px;color:var(--dim)">usado ${{v.used}}x</span>
      </div>
      ${{v.description?`<div style="font-size:10px;color:var(--dim);margin-bottom:6px">${{v.description}}</div>`:''}}
      <div style="display:flex;gap:5px">
        <button class="bsm" onclick="useVoice('${{v.engine}}','${{JSON.stringify(v.params).replace(/'/g,"\\'")}}'  )">Usar</button>
        <button class="bsm danger" onclick="deleteVoice('${{v.id}}')">Remover</button>
      </div>`;
    div.appendChild(el);
  }});
}}

async function saveVoiceToBank(){{
  const name=document.getElementById('vb-name').value.trim();
  if(!name){{setStatus('vb-status','err','Digite um nome');return;}}
  let params={{}};
  if(curEngine==='kokoro')params={{voice:curKVoice,speed:parseFloat(document.getElementById('k-speed').value||'1')}};
  else if(curEngine==='edge')params={{voice:curEdgeVoice,rate:document.getElementById('edge-rate').value,pitch:document.getElementById('edge-pitch').value}};
  else if(curEngine==='chatterbox')params={{exaggeration:parseFloat(document.getElementById('cb-exag').value||'0.5')}};
  const d=await fetch('/api/voices/saved',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{name,description:document.getElementById('vb-desc').value,engine:curEngine,params,sample_path:lastResult?.path||''}})}}).then(r=>r.json());
  setStatus('vb-status','ok','✅ "'+d.name+'" salva!');
  document.getElementById('vb-name').value='';document.getElementById('vb-desc').value='';
  loadVoiceBank();
}}

function saveToBank(){{
  if(!lastResult){{alert('Gere um áudio primeiro');return;}}
  const name=prompt('Nome para salvar essa voz:','Voz '+curEngine);
  if(!name)return;
  let params={{}};
  if(curEngine==='kokoro')params={{voice:curKVoice}};
  else if(curEngine==='edge')params={{voice:curEdgeVoice}};
  fetch('/api/voices/saved',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{name,engine:curEngine,params,sample_path:lastResult.path||''}})}})
    .then(r=>r.json()).then(d=>{{alert('✅ "'+d.name+'" salva no banco!');loadVoiceBank();}});
}}

function useVoice(engine,paramsStr){{
  try{{
    curEngine=engine;
    const params=JSON.parse(paramsStr);
    // switch engine tab
    document.querySelectorAll('.etab').forEach(b=>b.classList.remove('active','kokoro','f5','chat','edge'));
    const tab=document.querySelector(`.etab.${{engine==='chatterbox'?'chat':engine}}`);
    if(tab)tab.classList.add('active',engine==='chatterbox'?'chat':engine);
    document.querySelectorAll('.epanel').forEach(p=>p.classList.remove('active'));
    const epid=engine==='chatterbox'?'ep-chatterbox':'ep-'+engine;
    document.getElementById(epid)?.classList.add('active');
    if(engine==='kokoro'&&params.voice){{
      curKVoice=params.voice;
      document.querySelectorAll('#kvg .vbtn').forEach(b=>{{
        b.classList.remove('sel');
        if(b.querySelector('.vn')?.textContent===KV[params.voice]?.label)b.classList.add('sel');
      }});
    }}
    if(engine==='edge'&&params.voice){{curEdgeVoice=params.voice;document.getElementById('edge-voice').value=params.voice;}}
    // nav to gen tab
    document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
    document.getElementById('page-gen').classList.add('active');
    document.querySelectorAll('.nb').forEach(b=>b.classList.remove('active'));
    document.querySelector('.nb').classList.add('active');
  }}catch(e){{console.error(e);}}
}}

async function deleteVoice(id){{
  if(!confirm('Remover?'))return;
  await fetch('/api/voices/saved/'+id,{{method:'DELETE'}});loadVoiceBank();
}}

// ── HISTORY ──────────────────────────────────────────────────────────────────
async function loadHistory(q=''){{
  const url=q?`/api/history?q=${{encodeURIComponent(q)}}`:'/api/history';
  const data=await fetch(url).then(r=>r.json()).catch(()=>[]);
  const div=document.getElementById('h-list');div.innerHTML='';
  if(!data.length){{div.innerHTML='<div style="color:var(--dim);font-size:12px">Nenhum áudio ainda.</div>';return;}}
  data.forEach(h=>{{
    const el=document.createElement('div');el.className='hitem';
    const file=h.path?.split(/[\\/\\\\]/).pop()||'';
    el.innerHTML=`<span class="hbadge ${{h.engine||''}}">${{h.engine||'?'}}</span><span class="htxt">${{(h.text||'').substring(0,80)}}</span><span style="font-size:10px;color:var(--dim);flex-shrink:0">${{h.audio_s||0}}s</span>${{file?`<a class="bsm" href="/api/audio/${{file}}" download>💾</a>`:''}}`;
    div.appendChild(el);
  }});
}}
function exportHistory(){{window.open('/api/history/export');}}

// ── PROJECTS ─────────────────────────────────────────────────────────────────
async function loadProjects(){{
  const data=await fetch('/api/projects').then(r=>r.json()).catch(()=>[]);
  const div=document.getElementById('pj-list');div.innerHTML='';
  if(!data.length){{div.innerHTML='<div style="color:var(--dim);font-size:12px">Nenhum projeto.</div>';return;}}
  data.forEach(name=>{{
    const el=document.createElement('div');el.style.cssText='display:flex;align-items:center;gap:8px;margin-bottom:6px';
    el.innerHTML=`<span style="flex:1;font-size:12px">📁 ${{name}}</span><button class="bsm" onclick="loadProject('${{name}}')">Carregar</button><button class="bsm danger" onclick="deleteProject('${{name}}')">✕</button>`;
    div.appendChild(el);
  }});
}}
async function saveProject(){{
  const name=document.getElementById('pj-name').value.trim();if(!name)return;
  const data={{engine:curEngine,voice:curKVoice,edge_voice:curEdgeVoice,text:document.getElementById('g-text').value,last_file:lastResult?.path||''}};
  await fetch('/api/projects',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{name,data}})}});
  loadProjects();
}}
async function loadProject(name){{
  const d=await fetch(`/api/projects/${{name}}`).then(r=>r.json());
  if(d.text)document.getElementById('g-text').value=d.text;
  if(d.engine){{curEngine=d.engine;}}
}}
async function deleteProject(name){{
  if(!confirm('Remover projeto "'+name+'"?'))return;
  await fetch('/api/projects/'+encodeURIComponent(name),{{method:'DELETE'}});loadProjects();
}}

// ── PRONUNCIATION ─────────────────────────────────────────────────────────────
async function loadPron(){{
  const data=await fetch('/api/pronunciations').then(r=>r.json()).catch(()=>({{}}));
  const div=document.getElementById('pr-list');div.innerHTML='';
  if(!Object.keys(data).length){{div.innerHTML='<div style="color:var(--dim);font-size:12px">Dicionário vazio.</div>';return;}}
  Object.entries(data).forEach(([w,p])=>{{
    const el=document.createElement('div');el.style.cssText='display:flex;align-items:center;gap:8px;margin-bottom:6px';
    el.innerHTML=`<span style="flex:1;font-size:12px"><b>${{w}}</b> → ${{p}}</span><button class="bsm danger" onclick="delPron('${{w}}')">✕</button>`;
    div.appendChild(el);
  }});
}}
async function addPron(){{
  const w=document.getElementById('pr-word').value.trim(),p=document.getElementById('pr-as').value.trim();
  if(!w||!p)return;
  await fetch('/api/pronunciations',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify({{word:w,phonetic:p}})}});
  document.getElementById('pr-word').value='';document.getElementById('pr-as').value='';loadPron();
}}
async function delPron(w){{await fetch('/api/pronunciations/'+encodeURIComponent(w),{{method:'DELETE'}});loadPron();}}

// ── STATS ─────────────────────────────────────────────────────────────────────
async function loadStats(){{
  const d=await fetch('/api/stats').then(r=>r.json()).catch(()=>({{}}));
  const cards=[
    {{n:d.total_generated||0,l:'Áudios Gerados'}},{{n:d.total_chars||0,l:'Chars Processados'}},
    {{n:(d.total_seconds||0)+'s',l:'Áudio Gerado'}},{{n:d.history_count||0,l:'No Histórico'}},
    {{n:d.projects||0,l:'Projetos'}},{{n:'$0.00',l:'Custo Total'}},
  ];
  const div=document.getElementById('st-cards');div.innerHTML='';
  cards.forEach(c=>{{div.innerHTML+=`<div class="sc"><div class="sn">${{c.n}}</div><div class="sl">${{c.l}}</div></div>`;}});
  const chars=d.total_chars||0;
  const cost=(chars/1000*0.30).toFixed(4);
  document.getElementById('st-cost').innerHTML=`<div style="font-size:12px;line-height:2.2">ElevenLabs: <b style="color:var(--r)">$${{cost}}</b><br>QWN3-TTS: <b style="color:var(--g)">$0.00</b><br>Economia: <b style="color:var(--a3)">$${{cost}} (100%)</b></div>`;
}}
</script>
</body>
</html>""".replace("{{kv}}",kv).replace("{{ev}}",ev).replace("{{eq}}",eq).replace("{{er}}",er).replace("{{ep}}",ep)

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  QWN3-TTS STUDIO v5.0 — Multi-Engine Production")
    print(f"  GPU: {GPU_NAME} {GPU_VRAM}")
    print("  Engines: Kokoro · F5-TTS · Chatterbox · Edge TTS")
    print("  http://localhost:7860")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="warning",
                timeout_keep_alive=300)

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

try:
    import pyloudnorm as pyln; PYLN_OK = True
except ImportError:
    PYLN_OK = False

try:
    import resampy; RESAMPY_OK = True
except ImportError:
    RESAMPY_OK = False

try:
    import noisereduce as nr; NR_OK = True
except ImportError:
    NR_OK = False

try:
    from faster_whisper import WhisperModel; WHISPER_OK = True
except ImportError:
    WHISPER_OK = False

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

def _load_xtts():
    with _engine_lock:
        if "xtts" not in _engines:
            from TTS.api import TTS
            _engines["xtts"] = TTS("tts_models/multilingual/multi-dataset/xtts_v2",
                                   gpu=CUDA)
    return _engines["xtts"]

def _load_df():
    with _engine_lock:
        if "df" not in _engines:
            from df.enhance import init_df
            model, df_state, _ = init_df()
            _engines["df"] = (model, df_state)
    return _engines["df"]

def _load_resemble():
    with _engine_lock:
        if "resemble" not in _engines:
            import torch
            from resemble_enhance.enhancer.inference import load_enhancer
            _engines["resemble"] = load_enhancer(torch.device(DEVICE))
    return _engines["resemble"]

_whisper_model = None
_whisper_lock  = threading.Lock()

def transcribe_audio(path: str) -> str:
    global _whisper_model
    if not WHISPER_OK: return ""
    try:
        with _whisper_lock:
            if _whisper_model is None:
                _whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        segs, _ = _whisper_model.transcribe(path, beam_size=5)
        return " ".join(s.text for s in segs).strip()
    except Exception:
        return ""

XTTS_SPEAKERS = [
    "Ana Florence","Claribel Dervla","Daisy Studious","Gracie Wise",
    "Tammie Ema","Alison Dietlinde","Nova Hogarth","Damien Black",
    "Aaron Dreschner","Craig Gutsy","Baldur Sanjin","Viktor Eka",
]
XTTS_LANGS = ["pt","en","es","fr","de","it","pl","tr","ru","nl","cs","ar","zh-cn","ko","ja","hi"]

# Preload Chatterbox (best cloning) + Kokoro (fastest) in background
threading.Thread(target=_load_chatterbox, daemon=True).start()
threading.Thread(target=_load_kokoro,     daemon=True).start()

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

# ── Broadcast / Professional Mastering ───────────────────────────────────────
def fx_lufs_normalize(w: np.ndarray, sr: int, target: float = -14.0) -> np.ndarray:
    if not PYLN_OK: return fx_normalize(w, target + 3)
    try:
        meter   = pyln.Meter(sr)
        w2      = w.reshape(-1, 1) if w.ndim == 1 else w
        lufs    = meter.integrated_loudness(w2.astype(float))
        if not np.isfinite(lufs): return w
        return pyln.normalize.loudness(w2, lufs, target).squeeze().astype(np.float32)
    except Exception:
        return fx_normalize(w)

def fx_air_eq(w: np.ndarray, sr: int, gain_db: float = 3.0, freq: int = 12000) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    fc = min(freq, sr // 2 - 1)
    sos = butter(2, fc, btype="high", fs=sr, output="sos")
    hi  = sosfilt(sos, w.astype(np.float32))
    return np.clip(w + hi * (10 ** (gain_db / 20) - 1), -1.0, 1.0)

def fx_multiband_compress(w: np.ndarray, sr: int) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    def band(lo, hi, t):
        if lo is None: sos = butter(2, hi, btype="low",  fs=sr, output="sos")
        elif hi is None: sos = butter(2, lo, btype="high", fs=sr, output="sos")
        else:            sos = butter(2, [lo, hi], btype="band", fs=sr, output="sos")
        x = sosfilt(sos, w.astype(np.float32))
        m = np.abs(x) > t
        x[m] = np.sign(x[m]) * (t + (np.abs(x[m]) - t) / 3.5)
        return x
    return np.clip(band(None, 300, 0.55) + band(300, 4000, 0.65) + band(4000, None, 0.75), -1, 1).astype(np.float32)

def fx_true_peak_limit(w: np.ndarray, ceiling: float = 0.985) -> np.ndarray:
    peak = np.max(np.abs(w))
    return w / peak * ceiling if peak > ceiling else w

def fx_hq_resample(w: np.ndarray, sr: int, target: int = 48000) -> tuple:
    if sr == target: return w, sr
    if RESAMPY_OK:
        return resampy.resample(w.astype(np.float32), sr, target), target
    from scipy.signal import resample_poly
    from math import gcd; g = gcd(sr, target)
    return resample_poly(w, target // g, sr // g).astype(np.float32), target

def fx_noisereduce(w: np.ndarray, sr: int) -> np.ndarray:
    if NR_OK:
        try: return nr.reduce_noise(y=w, sr=sr, stationary=False).astype(np.float32)
        except Exception: pass
    return fx_denoise(w, sr)

def fx_neural_enhance(w: np.ndarray, sr: int) -> tuple:
    try:
        import torch
        from resemble_enhance.enhancer.inference import enhance as re_enh
        tmp = tempfile.mktemp(suffix=".wav")
        sf.write(tmp, w.astype(np.float32), sr)
        enh, new_sr = re_enh(tmp, torch.device(DEVICE))
        os.unlink(tmp)
        return enh.cpu().numpy().astype(np.float32), int(new_sr)
    except Exception:
        return w, sr

def fx_deep_denoise(w: np.ndarray, sr: int) -> np.ndarray:
    try:
        import torch
        from df.enhance import enhance as df_enh, init_df
        model, df_state, _ = init_df()
        audio = torch.from_numpy(w.reshape(1, -1)).float()
        return df_enh(model, df_state, audio).numpy().squeeze().astype(np.float32)
    except Exception:
        return fx_noisereduce(w, sr)

# Quality presets
QUALITY_PRESETS = {
    "Rápido":       {"normalize": True},
    "Padrão":       {"normalize": True, "trim": True, "compress": True},
    "Profissional": {"normalize": True, "trim": True, "compress": True,
                     "lufs": True, "air_eq": True, "fade": True},
    "Broadcast":    {"normalize": True, "trim": True, "multiband": True,
                     "lufs": True, "air_eq": True, "limiter": True, "hq_sr": True},
    "Cinema":       {"normalize": True, "trim": True, "compress": True,
                     "lufs": True, "air_eq": True, "reverb": True, "eq": "Cinema"},
    "Podcast":      {"normalize": True, "trim": True, "compress": True,
                     "lufs": True, "eq": "Podcast", "fade": True},
}

def fx_pipeline(w, sr, opts):
    # apply preset if set
    preset_name = opts.get("preset", "")
    if preset_name and preset_name in QUALITY_PRESETS:
        opts = {**QUALITY_PRESETS[preset_name], **opts}

    if opts.get("normalize"):           w = fx_normalize(w)
    if opts.get("trim"):                w = fx_trim(w, sr)
    speed = float(opts.get("speed", 1.0))
    if abs(speed - 1.0) > 0.01:        w, sr = fx_speed(w, sr, speed)
    pitch = float(opts.get("pitch", 0.0))
    if abs(pitch) > 0.1:               w = fx_pitch(w, sr, pitch)
    if opts.get("deep_denoise"):        w = fx_deep_denoise(w, sr)
    elif opts.get("noisereduce"):       w = fx_noisereduce(w, sr)
    elif opts.get("denoise"):           w = fx_denoise(w, sr)
    if opts.get("multiband"):           w = fx_multiband_compress(w, sr)
    elif opts.get("compress"):          w = fx_compress(w)
    eq = opts.get("eq", "Neutro")
    if eq and eq != "Neutro":
        p = EQ_PRESETS.get(eq) or {}
        if p: w = fx_eq(w, sr, p.get("bass",0), p.get("mid",0), p.get("treble",0))
    if opts.get("reverb"):              w = fx_reverb(w, sr, float(opts.get("reverb_amount", 0.3)))
    if opts.get("echo"):                w = fx_echo(w, sr)
    if opts.get("air_eq"):              w = fx_air_eq(w, sr)
    if opts.get("lufs"):                w = fx_lufs_normalize(w, sr, float(opts.get("lufs_target", -14.0)))
    if opts.get("fade"):                w = fx_fade(w, sr)
    if opts.get("padding"):             w = fx_padding(w, sr)
    if opts.get("limiter"):             w = fx_true_peak_limit(w)
    else:                               w = fx_normalize(w)
    # Post-pipeline
    if opts.get("neural_enhance"):      w, sr = fx_neural_enhance(w, sr)
    if opts.get("hq_sr"):               w, sr = fx_hq_resample(w, sr, 48000)
    return w, sr

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

# ── Engine: XTTS v2 ──────────────────────────────────────────────────────────
def gen_xtts(text: str, language: str = "pt", speaker: str = "Ana Florence",
             speaker_wav: str = None, opts: dict = None) -> tuple:
    opts  = opts or {}
    model = _load_xtts()
    text  = fx_apply_pron(fx_clean_text(text))
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    try:
        if speaker_wav and Path(speaker_wav).exists():
            model.tts_to_file(text=text, speaker_wav=speaker_wav,
                              language=language, file_path=tmp)
        else:
            model.tts_to_file(text=text, speaker=speaker,
                              language=language, file_path=tmp)
        w, sr = sf.read(tmp)
        w = w.astype(np.float32)
        if w.ndim > 1: w = w.mean(-1)
        w, sr = fx_pipeline(w, int(sr), opts)
        return w, int(sr)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

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

    elif engine == "xtts":
        lang    = params.get("language", "pt")
        speaker = params.get("speaker", "Ana Florence")
        ref     = params.get("ref_audio")
        w, sr   = gen_xtts(text, lang, speaker, ref, opts)
        used_voice = f"xtts-{speaker.split()[0].lower()}"

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
    return {
        "gpu": GPU_NAME, "vram": GPU_VRAM, "version": "6.0",
        "engines_loaded": list(_engines.keys()),
        "engines_available": ["kokoro","f5","chatterbox","edge","xtts"],
        "capabilities": {
            "whisper": WHISPER_OK,
            "pyloudnorm": PYLN_OK,
            "resampy": RESAMPY_OK,
            "noisereduce": NR_OK,
            "neural_enhance": "resemble" in _engines,
            "deep_denoise": "df" in _engines,
        },
        "quality_presets": list(QUALITY_PRESETS.keys()),
        "saved_voices": len(_saved_voices),
    }

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

@app.get("/api/quality_presets")
async def quality_presets(): return list(QUALITY_PRESETS.keys())

@app.get("/api/xtts/speakers")
async def xtts_speakers(): return XTTS_SPEAKERS

@app.get("/api/xtts/langs")
async def xtts_langs(): return XTTS_LANGS

@app.post("/api/transcribe")
async def api_transcribe(audio: UploadFile = File(...)):
    if not WHISPER_OK:
        return JSONResponse({"error": "faster-whisper não instalado"}, status_code=501)
    tmp = None
    try:
        data = await audio.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(data); tmp = f.name
        text = await asyncio.to_thread(transcribe_audio, tmp)
        return {"text": text, "ok": bool(text)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        if tmp and os.path.exists(tmp): os.unlink(tmp)

@app.post("/api/enhance")
async def api_enhance(request: Request):
    req      = await _body(request)
    filename = req.get("filename", "")
    path     = OUT_DIR / filename
    if not path.exists():
        return JSONResponse({"error": "Arquivo não encontrado"}, status_code=404)
    try:
        w, sr = sf.read(str(path))
        w = w.astype(np.float32)
        if w.ndim > 1: w = w.mean(-1)
        opts = req.get("opts", {"lufs": True, "air_eq": True, "noisereduce": True, "limiter": True})
        w, sr = fx_pipeline(w, sr, opts)
        enh_path = _save_wav(w, sr, "enhanced")
        return JSONResponse({
            "audio_b64": _wav_to_b64(w, sr),
            "filename": enh_path.name,
            "stats": _audio_stats(w, sr),
            "waveform": fx_waveform_data(w),
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# This file is merged into tts_studio.py — do not run directly
def _html():  # noqa: C901
    kv  = __import__("json").dumps(KOKORO_VOICES, ensure_ascii=False)
    ev  = __import__("json").dumps(EDGE_VOICES,   ensure_ascii=False)
    eq  = __import__("json").dumps(list(EQ_PRESETS.keys()), ensure_ascii=False)
    er  = __import__("json").dumps(EDGE_RATES, ensure_ascii=False)
    ep  = __import__("json").dumps(EDGE_PITCHES, ensure_ascii=False)
    gpu = f"{GPU_NAME} {GPU_VRAM}".strip()

    H = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QWN3 Studio — Professional Voice AI</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#020209;
  --s1:#080812;
  --s2:#0d0d1c;
  --s3:#121224;
  --s4:#1a1a30;
  --bd:rgba(255,255,255,0.07);
  --bd2:rgba(255,255,255,0.12);
  --t:#f1f5f9;
  --t2:#94a3b8;
  --t3:#475569;
  --ind:#818cf8;
  --ind2:#6366f1;
  --vio:#a78bfa;
  --cyn:#22d3ee;
  --em:#4ade80;
  --ros:#fb7185;
  --amb:#fbbf24;
  --kok:#818cf8;
  --f5c:#22d3ee;
  --cha:#4ade80;
  --edg:#fbbf24;
  --glow-ind:rgba(99,102,241,0.3);
  --glow-cyn:rgba(6,182,212,0.25);
}
html,body{height:100%;overflow:hidden;font-family:'Inter',system-ui,sans-serif;background:var(--bg);color:var(--t)}

/* ── BG ORBS ─────────────────────────────────────────── */
.orbs{position:fixed;inset:0;pointer-events:none;z-index:0;overflow:hidden}
.orb{position:absolute;border-radius:50%;filter:blur(90px)}
.orb1{width:900px;height:900px;top:-350px;left:-250px;
  background:radial-gradient(circle,rgba(99,102,241,0.13),transparent 65%);
  animation:o1 24s ease-in-out infinite alternate}
.orb2{width:650px;height:650px;bottom:-200px;right:-150px;
  background:radial-gradient(circle,rgba(6,182,212,0.10),transparent 65%);
  animation:o2 19s ease-in-out infinite alternate}
.orb3{width:500px;height:500px;top:45%;left:45%;
  background:radial-gradient(circle,rgba(139,92,246,0.08),transparent 65%);
  animation:o3 28s ease-in-out infinite alternate}
@keyframes o1{0%{transform:translate(0,0)}100%{transform:translate(70px,90px)}}
@keyframes o2{0%{transform:translate(0,0)}100%{transform:translate(-60px,-55px)}}
@keyframes o3{0%{transform:translate(-50%,-50%) scale(1)}100%{transform:translate(calc(-50% + 50px),calc(-50% - 40px)) scale(1.18)}}

/* ── LAYOUT ──────────────────────────────────────────── */
.shell{position:relative;z-index:1;display:flex;flex-direction:column;height:100vh}
header{height:58px;flex-shrink:0;display:flex;align-items:center;gap:16px;padding:0 20px;
  background:rgba(8,8,18,0.8);backdrop-filter:blur(20px);
  border-bottom:1px solid var(--bd)}
.logo{display:flex;align-items:center;gap:10px;text-decoration:none}
.logo-mark{width:34px;height:34px;background:linear-gradient(135deg,#6366f1,#22d3ee);
  border-radius:10px;display:flex;align-items:center;justify-content:center;
  font-size:16px;font-weight:900;color:#fff;flex-shrink:0;
  box-shadow:0 0 20px rgba(99,102,241,0.4)}
.logo-text{font-size:15px;font-weight:800;color:var(--t);letter-spacing:-0.3px}
.logo-ver{font-size:10px;padding:2px 7px;background:rgba(99,102,241,0.15);
  border:1px solid rgba(99,102,241,0.3);border-radius:20px;color:var(--ind);font-weight:600}
.hbadge{padding:4px 12px;background:rgba(255,255,255,0.04);border:1px solid var(--bd);
  border-radius:20px;font-size:11px;color:var(--t2);white-space:nowrap}
.hbadge.live{color:var(--em);border-color:rgba(74,222,128,0.3);background:rgba(74,222,128,0.06)}
.hspacer{flex:1}
.app{display:flex;flex:1;overflow:hidden}

/* ── SIDEBAR ─────────────────────────────────────────── */
nav.sidebar{width:224px;flex-shrink:0;background:rgba(8,8,18,0.6);
  backdrop-filter:blur(20px);border-right:1px solid var(--bd);
  display:flex;flex-direction:column;overflow-y:auto;padding:12px 0}
.nav-sec{padding:10px 16px 5px;font-size:9px;font-weight:700;letter-spacing:2px;
  text-transform:uppercase;color:var(--t3)}
.nav-item{display:flex;align-items:center;gap:10px;padding:9px 16px;margin:1px 8px;
  border-radius:9px;cursor:pointer;color:var(--t2);font-size:12.5px;font-weight:500;
  border:1px solid transparent;transition:all .15s;user-select:none}
.nav-item:hover{background:rgba(255,255,255,0.05);color:var(--t)}
.nav-item.active{background:rgba(99,102,241,0.12);border-color:rgba(99,102,241,0.25);
  color:var(--ind)}
.nav-item .ni{font-size:15px;width:18px;text-align:center;flex-shrink:0}
.nav-bottom{margin-top:auto;padding:12px 8px 4px;border-top:1px solid var(--bd)}

/* ── MAIN ────────────────────────────────────────────── */
main{flex:1;overflow-y:auto;padding:22px 24px}
.page{display:none;max-width:1100px;margin:0 auto}
.page.active{display:block;animation:pfade .2s ease}
@keyframes pfade{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
.ph{font-size:20px;font-weight:800;margin-bottom:2px;letter-spacing:-0.4px}
.ps{font-size:12px;color:var(--t2);margin-bottom:20px}

/* ── GRID LAYOUTS ────────────────────────────────────── */
.studio-grid{display:grid;grid-template-columns:380px 1fr;gap:18px}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}
.g4{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
@media(max-width:900px){.studio-grid{grid-template-columns:1fr}.g2,.g3,.g4{grid-template-columns:1fr 1fr}}

/* ── CARDS ───────────────────────────────────────────── */
.card{background:rgba(13,13,28,0.75);backdrop-filter:blur(16px);
  border:1px solid var(--bd);border-radius:14px;padding:18px;
  transition:border-color .2s}
.card:hover{border-color:var(--bd2)}
.card+.card{margin-top:14px}
.ct{font-size:10px;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;
  color:var(--t3);margin-bottom:14px}

/* ── ENGINE TILES ────────────────────────────────────── */
.eng-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:16px}
.eng-tile{background:rgba(255,255,255,0.03);border:1px solid var(--bd);
  border-radius:11px;padding:12px 14px;cursor:pointer;transition:all .18s;
  display:flex;flex-direction:column;gap:5px}
.eng-tile:hover{background:rgba(255,255,255,0.06);border-color:var(--bd2)}
.eng-tile.active-kok{background:rgba(99,102,241,0.12);border-color:rgba(99,102,241,0.4);
  box-shadow:0 0 20px rgba(99,102,241,0.12)}
.eng-tile.active-f5{background:rgba(6,182,212,0.10);border-color:rgba(6,182,212,0.4);
  box-shadow:0 0 20px rgba(6,182,212,0.10)}
.eng-tile.active-cha{background:rgba(74,222,128,0.08);border-color:rgba(74,222,128,0.35);
  box-shadow:0 0 20px rgba(74,222,128,0.08)}
.eng-tile.active-edg{background:rgba(251,191,36,0.08);border-color:rgba(251,191,36,0.35);
  box-shadow:0 0 20px rgba(251,191,36,0.08)}
.eng-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}
.eng-hd{display:flex;align-items:center;gap:7px;font-size:12px;font-weight:700;color:var(--t)}
.eng-sub{font-size:10px;color:var(--t3);padding-left:15px}

/* ── VOICE GRID ──────────────────────────────────────── */
.vg{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
.vc{background:rgba(255,255,255,0.03);border:1px solid var(--bd);border-radius:10px;
  padding:10px 8px;cursor:pointer;text-align:center;transition:all .15s;
  display:flex;flex-direction:column;align-items:center;gap:5px}
.vc:hover{background:rgba(255,255,255,0.06);border-color:var(--bd2)}
.vc.sel{border-color:var(--ind);background:rgba(99,102,241,0.12)}
.va{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;
  justify-content:center;font-size:11px;font-weight:800;flex-shrink:0}
.va.F{background:linear-gradient(135deg,#6366f1,#a78bfa)}
.va.M{background:linear-gradient(135deg,#0891b2,#22d3ee)}
.vn{font-size:11px;font-weight:600;color:var(--t)}
.vd{font-size:9px;color:var(--t3);line-height:1.3;text-align:center}
.vc.sel .vd{color:var(--ind)}

/* ── FORM ELEMENTS ───────────────────────────────────── */
label{font-size:11px;color:var(--t2);display:block;margin-bottom:5px;margin-top:12px;font-weight:500}
label:first-child{margin-top:0}
select,input[type=text],input[type=number]{width:100%;background:rgba(255,255,255,0.04);
  border:1px solid var(--bd);color:var(--t);padding:9px 12px;border-radius:9px;
  font-size:12px;font-family:inherit;outline:none;transition:border .15s;appearance:none}
select:focus,input:focus{border-color:var(--ind)}
textarea{width:100%;background:rgba(255,255,255,0.03);border:1px solid var(--bd);
  color:var(--t);padding:14px;border-radius:11px;font-size:13px;font-family:inherit;
  outline:none;resize:vertical;min-height:180px;line-height:1.7;transition:border .15s}
textarea:focus{border-color:rgba(99,102,241,0.5);background:rgba(255,255,255,0.04)}
textarea::placeholder{color:var(--t3)}
input[type=range]{width:100%;accent-color:var(--ind);cursor:pointer;height:4px;border-radius:4px;
  background:var(--s4);border:none;padding:0;outline:none}

/* ── SLIDER ROW ──────────────────────────────────────── */
.srow{display:flex;align-items:center;gap:10px;margin-top:4px}
.sv{font-size:11px;color:var(--ind);min-width:38px;text-align:right;font-weight:600}

/* ── TOGGLES ─────────────────────────────────────────── */
.tog-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px}
.tog{display:flex;align-items:center;gap:8px;background:rgba(255,255,255,0.03);
  border:1px solid var(--bd);border-radius:9px;padding:9px 11px;cursor:pointer;
  font-size:11px;color:var(--t2);transition:all .15s;user-select:none}
.tog:hover{background:rgba(255,255,255,0.06);color:var(--t)}
.tog.on{border-color:rgba(99,102,241,0.4);background:rgba(99,102,241,0.10);color:var(--ind)}
.tdot{width:7px;height:7px;border-radius:50%;background:var(--s4);flex-shrink:0;transition:.15s}
.tog.on .tdot{background:var(--ind)}

/* ── BUTTONS ─────────────────────────────────────────── */
.gen-btn{width:100%;padding:15px;border:none;border-radius:12px;
  background:linear-gradient(135deg,#4f46e5 0%,#7c3aed 50%,#0e7490 100%);
  background-size:200% 200%;animation:gbg 5s ease infinite;
  color:#fff;font-size:14px;font-weight:700;cursor:pointer;
  position:relative;overflow:hidden;letter-spacing:.4px;margin-top:14px;
  font-family:inherit;transition:transform .2s,box-shadow .2s}
.gen-btn:hover{transform:translateY(-2px);box-shadow:0 10px 35px rgba(99,102,241,0.35)}
.gen-btn:active{transform:translateY(0)}
.gen-btn:disabled{opacity:.45;cursor:default;animation:none;transform:none;box-shadow:none}
.gen-btn::after{content:'';position:absolute;inset:0;
  background:linear-gradient(105deg,transparent 33%,rgba(255,255,255,0.14) 50%,transparent 67%);
  animation:shimmer 3.5s ease-in-out infinite}
@keyframes gbg{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
@keyframes shimmer{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}

.btn-sm{background:rgba(255,255,255,0.05);border:1px solid var(--bd);color:var(--t2);
  padding:7px 14px;border-radius:8px;cursor:pointer;font-size:11px;font-weight:500;
  font-family:inherit;transition:all .15s}
.btn-sm:hover{background:rgba(255,255,255,0.09);border-color:var(--bd2);color:var(--t)}
.btn-sm.primary{background:rgba(99,102,241,0.15);border-color:rgba(99,102,241,0.4);color:var(--ind)}
.btn-sm.primary:hover{background:rgba(99,102,241,0.25)}
.btn-sm.danger{background:rgba(251,113,133,0.08);border-color:rgba(251,113,133,0.3);color:var(--ros)}
.btn-sm.danger:hover{background:rgba(251,113,133,0.16)}
.brow{display:flex;gap:7px;flex-wrap:wrap;margin-top:10px}

/* ── STATUS / TOAST ──────────────────────────────────── */
.status{display:none;align-items:center;gap:9px;padding:10px 14px;border-radius:10px;
  font-size:12px;margin-top:10px}
.status.show{display:flex}
.status.ok{background:rgba(74,222,128,0.08);border:1px solid rgba(74,222,128,0.25);color:var(--em)}
.status.err{background:rgba(251,113,133,0.08);border:1px solid rgba(251,113,133,0.25);color:var(--ros)}
.status.load{background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.25);color:var(--ind)}
.spin{width:14px;height:14px;border:2px solid rgba(99,102,241,0.25);
  border-top-color:var(--ind);border-radius:50%;animation:sp .7s linear infinite;flex-shrink:0}
@keyframes sp{to{transform:rotate(360deg)}}

#toasts{position:fixed;bottom:24px;right:24px;z-index:9999;
  display:flex;flex-direction:column;gap:8px;pointer-events:none}
.toast{background:rgba(13,13,28,0.95);backdrop-filter:blur(20px);
  border:1px solid var(--bd2);border-radius:11px;padding:12px 16px;
  font-size:12px;color:var(--t);display:flex;align-items:center;gap:9px;
  min-width:220px;max-width:320px;pointer-events:auto;
  animation:tin .3s ease;box-shadow:0 8px 32px rgba(0,0,0,0.5)}
.toast.out{animation:tout .3s ease forwards}
@keyframes tin{from{opacity:0;transform:translateX(40px)}to{opacity:1;transform:none}}
@keyframes tout{to{opacity:0;transform:translateX(40px)}}

/* ── AUDIO PLAYER ────────────────────────────────────── */
.player-card{background:rgba(13,13,28,0.85);border:1px solid var(--bd);
  border-radius:14px;padding:18px;margin-top:14px;display:none}
.player-card.show{display:block;animation:pfade .25s ease}
audio{width:100%;height:36px;border-radius:8px;margin-bottom:10px;outline:none;
  filter:invert(0.9) hue-rotate(180deg) saturate(0.8)}
canvas#wv{width:100%;height:64px;border-radius:10px;background:rgba(255,255,255,0.02);
  cursor:pointer;display:block}
.ameta{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-top:12px}
.am{background:rgba(255,255,255,0.03);border:1px solid var(--bd);border-radius:9px;
  padding:9px;text-align:center}
.am-v{font-size:15px;font-weight:700;color:var(--t)}
.am-l{font-size:9px;color:var(--t3);text-transform:uppercase;letter-spacing:1px;margin-top:2px}

/* ── WAVEFORM LOADING ────────────────────────────────── */
.wv-loader{width:100%;height:64px;border-radius:10px;background:rgba(255,255,255,0.02);
  display:flex;align-items:center;justify-content:center;gap:3px;padding:8px}
.wvb{width:4px;border-radius:3px;background:linear-gradient(to top,var(--ind2),var(--cyn));
  animation:wvp 1s ease-in-out infinite}
@keyframes wvp{0%,100%{transform:scaleY(.15)}50%{transform:scaleY(1)}}

/* ── UPLOAD ZONE ─────────────────────────────────────── */
.upzone{border:2px dashed var(--bd);border-radius:11px;padding:28px;
  text-align:center;cursor:pointer;position:relative;transition:all .2s}
.upzone:hover,.upzone.drag{border-color:var(--ind);background:rgba(99,102,241,0.05)}
.upzone.has{border-color:var(--em);background:rgba(74,222,128,0.05)}
.upzone input{position:absolute;inset:0;opacity:0;cursor:pointer}
.upzone .up-ico{font-size:30px;display:block;margin-bottom:8px}
.upzone p{font-size:12px;color:var(--t2)}

/* ── CHAR COUNT ──────────────────────────────────────── */
.cbar{display:flex;justify-content:space-between;font-size:10px;color:var(--t3);margin-top:5px}
.cbar span{color:var(--ind);font-weight:600}

/* ── HISTORY ITEM ────────────────────────────────────── */
.hi{background:rgba(255,255,255,0.03);border:1px solid var(--bd);border-radius:10px;
  padding:11px 14px;display:flex;align-items:center;gap:10px;margin-bottom:7px;
  transition:border-color .15s}
.hi:hover{border-color:var(--bd2)}
.htxt{flex:1;font-size:11px;color:var(--t2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hbdg{padding:2px 8px;border-radius:10px;font-size:9px;font-weight:600;flex-shrink:0}
.hbdg.kokoro{background:rgba(99,102,241,0.15);color:var(--ind)}
.hbdg.f5{background:rgba(6,182,212,0.15);color:var(--cyn)}
.hbdg.chatterbox{background:rgba(74,222,128,0.12);color:var(--em)}
.hbdg.edge{background:rgba(251,191,36,0.12);color:var(--amb)}

/* ── STAT CARD ───────────────────────────────────────── */
.sc{background:rgba(255,255,255,0.03);border:1px solid var(--bd);border-radius:12px;
  padding:18px;text-align:center}
.sn{font-size:28px;font-weight:800;
  background:linear-gradient(135deg,var(--ind),var(--cyn));
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px}
.sl{font-size:10px;color:var(--t3);text-transform:uppercase;letter-spacing:1.2px}

/* ── ENGINE PANEL ────────────────────────────────────── */
.ep{display:none}.ep.act{display:block}

/* ── SEARCH ──────────────────────────────────────────── */
.search-box{width:100%;background:rgba(255,255,255,0.04);border:1px solid var(--bd);
  color:var(--t);padding:9px 14px;border-radius:9px;font-size:12px;
  font-family:inherit;outline:none;margin-bottom:14px;transition:border .15s}
.search-box:focus{border-color:var(--ind)}

/* ── SCROLLBAR ───────────────────────────────────────── */
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-thumb{background:rgba(255,255,255,0.1);border-radius:4px}
::-webkit-scrollbar-track{background:transparent}

/* ── BATCH TABLE ─────────────────────────────────────── */
.bt-row{display:flex;align-items:center;gap:8px;padding:8px 12px;
  background:rgba(255,255,255,0.03);border:1px solid var(--bd);
  border-radius:9px;margin-bottom:6px;font-size:11px}
.bt-ok{color:var(--em)}.bt-err{color:var(--ros)}

/* ── COST BANNER ─────────────────────────────────────── */
.cost-banner{background:linear-gradient(135deg,rgba(99,102,241,0.08),rgba(6,182,212,0.06));
  border:1px solid rgba(99,102,241,0.2);border-radius:11px;padding:14px 16px;
  display:none;margin-top:10px}
.cost-banner.show{display:block;animation:pfade .2s ease}
</style>
</head>
<body>
<div class="orbs">
  <div class="orb orb1"></div>
  <div class="orb orb2"></div>
  <div class="orb orb3"></div>
</div>
<div class="shell">

<!-- HEADER -->
<header>
  <div class="logo">
    <div class="logo-mark">Q</div>
    <span class="logo-text">QWN3 STUDIO</span>
    <span class="logo-ver">v5.0</span>
  </div>
  <div class="hspacer"></div>
  <div class="hbadge live" id="b-eng">carregando engines...</div>
  <div class="hbadge" id="b-gpu">__GPU__</div>
</header>

<div class="app">
<!-- SIDEBAR -->
<nav class="sidebar">
  <div class="nav-sec">Gerar</div>
  <div class="nav-item active" data-page="studio">
    <span class="ni">⚡</span>Studio
  </div>
  <div class="nav-item" data-page="clone">
    <span class="ni">🔬</span>Clonar Voz
  </div>
  <div class="nav-item" data-page="batch">
    <span class="ni">🚀</span>Batch
  </div>
  <div class="nav-item" data-page="podcast">
    <span class="ni">🎙</span>Podcast
  </div>
  <div class="nav-sec">Gerenciar</div>
  <div class="nav-item" data-page="vbank">
    <span class="ni">🗄️</span>Banco de Vozes
  </div>
  <div class="nav-item" data-page="history">
    <span class="ni">📋</span>Histórico
  </div>
  <div class="nav-item" data-page="projects">
    <span class="ni">📁</span>Projetos
  </div>
  <div class="nav-item" data-page="pron">
    <span class="ni">📖</span>Pronúncia
  </div>
  <div class="nav-sec">Análise</div>
  <div class="nav-item" data-page="stats">
    <span class="ni">📊</span>Analytics
  </div>
  <div class="nav-item" data-page="api">
    <span class="ni">🔌</span>API Docs
  </div>
  <div class="nav-bottom">
    <div style="font-size:10px;color:var(--t3);text-align:center">
      Sem limites · $0.00 · 100% local
    </div>
  </div>
</nav>

<main>

<!-- ══ STUDIO ══════════════════════════════════════════ -->
<div id="page-studio" class="page active">
  <div class="ph">⚡ Studio</div>
  <div class="ps">Síntese profissional · 4 engines · Sem limite de tamanho</div>
  <div class="studio-grid">

    <!-- LEFT PANEL -->
    <div>
      <div class="card">
        <div class="ct">Engine de Voz</div>
        <div class="eng-grid">
          <div class="eng-tile active-cha eng-flag" data-eng="chatterbox" onclick="setEng(this,'chatterbox')"
            style="grid-column:1/-1;background:linear-gradient(135deg,rgba(74,222,128,0.12),rgba(34,211,238,0.08));border-color:rgba(74,222,128,0.45);box-shadow:0 0 24px rgba(74,222,128,0.18)">
            <div class="eng-hd">
              <div class="eng-dot" style="background:var(--cha)"></div>
              <span>Chatterbox</span>
              <span style="margin-left:auto;font-size:9px;padding:2px 8px;background:rgba(74,222,128,0.2);border:1px solid rgba(74,222,128,0.4);border-radius:20px;color:var(--em);font-weight:700;letter-spacing:0.5px">⭐ BATE ELEVENLABS</span>
            </div>
            <div class="eng-sub">Resemble AI · SOTA cloning · Controle emocional · TTS Arena #1</div>
          </div>
          <div class="eng-tile" data-eng="kokoro" onclick="setEng(this,'kokoro')">
            <div class="eng-hd"><div class="eng-dot" style="background:var(--kok)"></div>Kokoro</div>
            <div class="eng-sub">Rápido · PT-BR · Local</div>
          </div>
          <div class="eng-tile" data-eng="xtts" onclick="setEng(this,'xtts')">
            <div class="eng-hd"><div class="eng-dot" style="background:#f472b6"></div>XTTS v2</div>
            <div class="eng-sub">17 idiomas · Coqui</div>
          </div>
          <div class="eng-tile" data-eng="f5" onclick="setEng(this,'f5')">
            <div class="eng-hd"><div class="eng-dot" style="background:var(--f5c)"></div>F5-TTS</div>
            <div class="eng-sub">Zero-shot rápido</div>
          </div>
          <div class="eng-tile" data-eng="edge" onclick="setEng(this,'edge')">
            <div class="eng-hd"><div class="eng-dot" style="background:var(--edg)"></div>Edge TTS</div>
            <div class="eng-sub">322 vozes · Offline</div>
          </div>
        </div>

        <!-- Kokoro params -->
        <div id="ep-kokoro" class="ep">
          <div class="ct" style="margin-bottom:10px">Voz Kokoro</div>
          <div class="vg" id="kvg"></div>
          <label>Velocidade</label>
          <div class="srow">
            <input type="range" id="k-spd" min="0.5" max="2" step="0.05" value="1"
              oninput="document.getElementById('k-sv').textContent=this.value+'x'">
            <span class="sv" id="k-sv">1.0x</span>
          </div>
        </div>

        <!-- F5 params -->
        <div id="ep-f5" class="ep">
          <div style="font-size:11px;color:var(--t2);margin-bottom:10px">
            Grave 5–30s de qualquer voz e clone instantaneamente
          </div>
          <div class="upzone" id="f5-uz">
            <input type="file" id="f5-ref" accept="audio/*" onchange="onUp(this,'f5-uz','f5-lbl')">
            <span class="up-ico">🎤</span>
            <p id="f5-lbl">Arraste o áudio de referência aqui</p>
          </div>
          <label>Transcrição do áudio (melhora muito a qualidade)</label>
          <textarea id="f5-rt" rows="2" placeholder="O que o áudio diz..."></textarea>
        </div>

        <!-- Chatterbox params — DEFAULT (best cloning) -->
        <div id="ep-chatterbox" class="ep act">
          <div style="font-size:11px;color:var(--em);margin-bottom:10px;padding:10px;background:rgba(74,222,128,0.08);border:1px solid rgba(74,222,128,0.2);border-radius:9px;line-height:1.5">
            <b>⭐ Resemble AI Chatterbox</b> — Modelo #1 no TTS Arena (vence ElevenLabs em testes cegos).
            MIT · Watermark PerTh · Controle emocional único.
          </div>
          <div class="upzone" id="cb-uz">
            <input type="file" id="cb-ref" accept="audio/*" onchange="onUp(this,'cb-uz','cb-lbl')">
            <span class="up-ico">🎭</span>
            <p id="cb-lbl">Referência de voz (opcional · 5–10s basta)</p>
          </div>
          <label>Exageração Emocional <span style="color:var(--t3);font-weight:400">(único no mercado)</span></label>
          <div class="srow">
            <input type="range" id="cb-ex" min="0" max="1" step="0.05" value="0.5"
              oninput="document.getElementById('cb-ev').textContent=this.value;updateExagLabel(this.value)">
            <span class="sv" id="cb-ev">0.5</span>
          </div>
          <div id="cb-ex-desc" style="font-size:10px;color:var(--t3);margin-top:3px">
            0.5 = neutro · 0.8+ = dramático · &lt;0.3 = monótono
          </div>
        </div>

        <!-- Edge params -->
        <div id="ep-edge" class="ep">
          <label>Voz Microsoft</label>
          <select id="edge-v"></select>
          <label>Velocidade</label>
          <select id="edge-r"></select>
          <label>Tom (pitch)</label>
          <select id="edge-p"></select>
        </div>

        <!-- XTTS v2 panel -->
        <div id="ep-xtts" class="ep">
          <div style="font-size:11px;color:var(--t2);margin-bottom:10px;padding:8px;background:rgba(244,114,182,0.08);border:1px solid rgba(244,114,182,0.2);border-radius:8px">
            ⭐ XTTS v2 — melhor modelo open-source para clonagem multilingual
          </div>
          <label>Idioma</label>
          <select id="xt-lang">
            <option value="pt">Português BR</option>
            <option value="en">English</option>
            <option value="es">Español</option>
            <option value="fr">Français</option>
            <option value="de">Deutsch</option>
            <option value="it">Italiano</option>
            <option value="ja">Japanese</option>
            <option value="ko">Korean</option>
            <option value="zh-cn">Chinese</option>
            <option value="ar">Arabic</option>
          </select>
          <label>Speaker (sem referência)</label>
          <select id="xt-speaker"></select>
          <label>Áudio de referência (opcional — clona essa voz)</label>
          <div class="upzone" id="xt-uz">
            <input type="file" id="xt-ref" accept="audio/*" onchange="onUp(this,'xt-uz','xt-lbl');autoTranscribe('xt-ref','xt-rt')">
            <span class="up-ico">🎤</span>
            <p id="xt-lbl">Áudio para clonar (5–30s ideal)</p>
          </div>
          <label>Transcrição <span id="xt-tr-spin" style="color:var(--t3)"></span></label>
          <textarea id="xt-rt" rows="2" placeholder="Auto-transcrito por Whisper..."></textarea>
        </div>
      </div>

      <!-- FX CARD -->
      <div class="card">
        <div class="ct">Processamento de Áudio</div>
        <label>Pitch (semitons)</label>
        <div class="srow">
          <input type="range" id="g-pitch" min="-6" max="6" step="0.5" value="0"
            oninput="document.getElementById('g-pv').textContent=this.value">
          <span class="sv" id="g-pv">0</span>
        </div>
        <label>Preset de Qualidade</label>
        <select id="g-preset" onchange="applyPreset(this.value)"
          style="border-color:rgba(244,114,182,0.4);color:var(--t)">
          <option value="">Manual</option>
          <option value="Rápido">⚡ Rápido</option>
          <option value="Padrão" selected>🎙 Padrão</option>
          <option value="Profissional">🎚 Profissional</option>
          <option value="Broadcast">📡 Broadcast (48kHz LUFS)</option>
          <option value="Cinema">🎬 Cinema</option>
          <option value="Podcast">🎙 Podcast</option>
        </select>
        <label>EQ Preset</label>
        <select id="g-eq"></select>
        <div class="tog-grid" style="margin-top:12px">
          <div class="tog" onclick="tog(this,'normalize')"><div class="tdot"></div>Normalizar</div>
          <div class="tog" onclick="tog(this,'trim')"><div class="tdot"></div>Cortar Silêncio</div>
          <div class="tog" onclick="tog(this,'reverb')"><div class="tdot"></div>Reverb</div>
          <div class="tog" onclick="tog(this,'echo')"><div class="tdot"></div>Echo</div>
          <div class="tog" onclick="tog(this,'denoise')"><div class="tdot"></div>Denoise</div>
          <div class="tog" onclick="tog(this,'compress')"><div class="tdot"></div>Compressão</div>
          <div class="tog" onclick="tog(this,'fade')"><div class="tdot"></div>Fade In/Out</div>
          <div class="tog" onclick="tog(this,'padding')"><div class="tdot"></div>Padding</div>
        </div>
        <div style="margin-top:12px" class="tog-grid">
          <div class="tog" onclick="tog(this,'export_mp3')"><div class="tdot"></div>Exportar MP3</div>
          <div class="tog" onclick="tog(this,'generate_srt')"><div class="tdot"></div>Legendas SRT</div>
        </div>
      </div>
    </div>

    <!-- RIGHT PANEL -->
    <div>
      <div class="card">
        <div class="ct">Texto</div>
        <textarea id="g-txt" placeholder="Escreva o texto para sintetizar...&#10;&#10;Ctrl+Enter para gerar."
          oninput="onTxt(this)"></textarea>
        <div class="cbar">
          <span id="g-words"></span>
          <span><span id="g-cnt" style="color:var(--ind)">0</span> chars</span>
        </div>
        <button class="gen-btn" id="g-btn" onclick="doGen()">⚡ GERAR ÁUDIO</button>
      </div>

      <div id="g-status" class="status"></div>

      <!-- PLAYER -->
      <div id="g-player" class="player-card">
        <audio id="g-audio" controls></audio>
        <div id="g-wv-wrap">
          <canvas id="wv" height="64"></canvas>
        </div>
        <div class="ameta" id="g-ameta"></div>
        <div class="brow">
          <a id="g-dl" class="btn-sm" download>💾 WAV</a>
          <a id="g-dl-mp3" class="btn-sm" style="display:none" download>🎵 MP3</a>
          <a id="g-dl-srt" class="btn-sm" style="display:none" download>📄 SRT</a>
          <button class="btn-sm primary" onclick="saveToBank()">🗄️ Salvar Voz</button>
          <button class="btn-sm" onclick="showCost()">💰 vs ElevenLabs</button>
        </div>
        <div id="g-cost" class="cost-banner"></div>
      </div>
    </div>
  </div>
</div>

<!-- ══ CLONE ═══════════════════════════════════════════ -->
<div id="page-clone" class="page">
  <div class="ph">🔬 Clonar Voz</div>
  <div class="ps">F5-TTS ou Chatterbox · Qualquer áudio · Zero treinamento</div>
  <div class="g2">
    <div class="card">
      <label>Engine de Clonagem</label>
      <select id="cl-eng">
        <option value="f5">F5-TTS — Máxima fidelidade</option>
        <option value="chatterbox">Chatterbox — Controle emocional</option>
      </select>
      <label>Áudio de referência</label>
      <div class="upzone" id="cl-uz">
        <input type="file" id="cl-ref" accept="audio/*" onchange="onUp(this,'cl-uz','cl-lbl')">
        <span class="up-ico">🎤</span>
        <p id="cl-lbl">Qualquer áudio · Sem limite de tamanho</p>
      </div>
      <label>Transcrição do áudio (melhora F5-TTS)</label>
      <textarea id="cl-rt" rows="2" placeholder="O que o áudio de referência diz..."></textarea>
      <label>Texto novo para sintetizar</label>
      <textarea id="cl-txt" rows="4" placeholder="Este texto será falado com a voz clonada..."></textarea>
      <button class="gen-btn" id="cl-btn" onclick="doClone()">🔬 CLONAR E GERAR</button>
    </div>
    <div>
      <div id="cl-status" class="status"></div>
      <div id="cl-player" class="player-card">
        <audio id="cl-audio" controls></audio>
        <div class="brow"><a id="cl-dl" class="btn-sm" download>💾 Baixar WAV</a></div>
      </div>
    </div>
  </div>
</div>

<!-- ══ BATCH ════════════════════════════════════════════ -->
<div id="page-batch" class="page">
  <div class="ph">🚀 Batch — Geração em Massa</div>
  <div class="ps">Centenas de áudios · Sem filas · Sem limites</div>
  <div class="g2">
    <div class="card">
      <label>Engine</label>
      <select id="bt-eng">
        <option value="kokoro">Kokoro</option>
        <option value="edge">Edge TTS</option>
      </select>
      <label>Voz Kokoro (se selecionado)</label>
      <select id="bt-v"></select>
      <label>Textos (um por linha)</label>
      <textarea id="bt-txt" rows="10" placeholder="Linha 1: primeiro áudio&#10;Linha 2: segundo áudio&#10;..."></textarea>
      <button class="gen-btn" id="bt-btn" onclick="doBatch()">🚀 GERAR TUDO</button>
    </div>
    <div>
      <div id="bt-status" class="status"></div>
      <div id="bt-results"></div>
    </div>
  </div>
</div>

<!-- ══ PODCAST ══════════════════════════════════════════ -->
<div id="page-podcast" class="page">
  <div class="ph">🎙 Podcast — Múltiplos Locutores</div>
  <div class="ps">Crie diálogos e podcasts com vozes diferentes</div>
  <div class="card">
    <div class="ct">Script</div>
    <div id="pod-lines"></div>
    <div class="brow">
      <button class="btn-sm primary" onclick="addPodLine()">+ Linha</button>
      <button class="btn-sm" onclick="doPodcast()">🎙 Gerar Podcast</button>
    </div>
  </div>
  <div id="pod-status" class="status" style="margin-top:12px"></div>
  <div id="pod-player" class="player-card">
    <audio id="pod-audio" controls></audio>
    <div class="brow"><a id="pod-dl" class="btn-sm" download>💾 Baixar</a></div>
  </div>
</div>

<!-- ══ VOICE BANK ═══════════════════════════════════════ -->
<div id="page-vbank" class="page">
  <div class="ph">🗄️ Banco de Vozes</div>
  <div class="ps">Salve e reutilize configurações de voz perfeitas</div>
  <div class="g2">
    <div class="card">
      <div class="ct">Nova Voz</div>
      <label>Nome da Voz</label>
      <input type="text" id="vb-name" placeholder="Ex: Narrador Documentário">
      <label>Descrição</label>
      <input type="text" id="vb-desc" placeholder="Quando usar...">
      <button class="btn-sm primary" style="width:100%;margin-top:12px" onclick="saveVoice()">
        💾 Salvar Voz Atual
      </button>
    </div>
    <div>
      <div class="ct">Vozes Salvas</div>
      <div id="vb-list"></div>
    </div>
  </div>
</div>

<!-- ══ HISTORY ══════════════════════════════════════════ -->
<div id="page-history" class="page">
  <div class="ph">📋 Histórico</div>
  <div class="ps">Todos os áudios gerados · Busca · Reexportar</div>
  <div style="margin-bottom:14px;display:flex;gap:10px;align-items:center">
    <input class="search-box" id="h-q" placeholder="Buscar no histórico..." oninput="loadHistory()"
      style="margin-bottom:0;max-width:320px">
    <a class="btn-sm" href="/api/history/export" download>📥 Exportar CSV</a>
  </div>
  <div id="h-list"></div>
</div>

<!-- ══ PROJECTS ═════════════════════════════════════════ -->
<div id="page-projects" class="page">
  <div class="ph">📁 Projetos</div>
  <div class="ps">Salve estados completos do estúdio — volte onde parou</div>
  <div class="g2">
    <div class="card">
      <div class="ct">Salvar Projeto</div>
      <label>Nome do Projeto</label>
      <input type="text" id="pj-name" placeholder="Meu Projeto">
      <button class="btn-sm primary" style="width:100%;margin-top:12px" onclick="saveProject()">
        💾 Salvar Projeto Atual
      </button>
    </div>
    <div>
      <div class="ct">Projetos Salvos</div>
      <div id="pj-list"></div>
    </div>
  </div>
</div>

<!-- ══ PRONUNCIAÇÃO ═════════════════════════════════════ -->
<div id="page-pron" class="page">
  <div class="ph">📖 Dicionário de Pronúncia</div>
  <div class="ps">Corrija pronúncias específicas de palavras técnicas ou nomes</div>
  <div class="g2">
    <div class="card">
      <label>Palavra</label>
      <input type="text" id="pr-w" placeholder="Ex: API">
      <label>Pronúncia fonética</label>
      <input type="text" id="pr-p" placeholder="Ex: A-P-I">
      <button class="btn-sm primary" style="width:100%;margin-top:12px" onclick="addPron()">
        + Adicionar
      </button>
    </div>
    <div>
      <div class="ct">Dicionário</div>
      <div id="pr-list"></div>
    </div>
  </div>
</div>

<!-- ══ STATS ════════════════════════════════════════════ -->
<div id="page-stats" class="page">
  <div class="ph">📊 Analytics</div>
  <div class="ps">Métricas de uso e economia vs serviços pagos</div>
  <div class="g3" id="st-cards"></div>
  <div class="card" style="margin-top:16px" id="st-cost"></div>
</div>

<!-- ══ API ══════════════════════════════════════════════ -->
<div id="page-api" class="page">
  <div class="ph">🔌 API Reference</div>
  <div class="ps">REST API local — integre com qualquer sistema</div>
  <div class="card">
    <div class="ct">Endpoints</div>
    <pre style="font-size:11px;color:var(--t2);line-height:2;background:var(--s3);padding:16px;border-radius:9px;overflow-x:auto">
POST /api/generate
  { "engine": "kokoro|f5|chatterbox|edge",
    "text": "...",
    "params": { "voice": "af_heart", "speed": 1.0 },
    "opts": { "normalize": true, "trim": true } }

POST /api/generate/clone
  multipart/form-data: text, engine, audio_ref (file), ref_text

POST /api/batch
  { "engine": "...", "texts": ["...", "..."], "params": {...} }

POST /api/podcast
  { "script": [{"text":"...","engine":"kokoro","voice":"af_heart"}] }

GET  /api/history          GET  /api/stats
GET  /api/voices/saved     POST /api/voices/saved
GET  /api/projects         POST /api/projects
GET  /api/pronunciations   POST /api/pronunciations
GET  /api/audio/{filename}
    </pre>
  </div>
</div>

</main>
</div><!-- .app -->
</div><!-- .shell -->

<div id="toasts"></div>

<script>
// ── DATA (injected by server) ─────────────────────────
const KV = __KV__;
const EV = __EV__;
const EQ = __EQ__;
const ER = __ER__;
const EP = __EP__;

// ── STATE ─────────────────────────────────────────────
let curEng = 'chatterbox';
let curKV  = 'af_heart';
let curEV  = 'pt-BR-FranciscaNeural';
let toggs  = {};
let lastRes = null;
let wvRaf = null;
let wvCtx = null;

// ── NAVIGATION ────────────────────────────────────────
document.querySelectorAll('.nav-item').forEach(el => {
  el.addEventListener('click', () => {
    document.querySelectorAll('.nav-item').forEach(x => x.classList.remove('active'));
    el.classList.add('active');
    const pg = el.dataset.page;
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.getElementById('page-'+pg).classList.add('active');
    if (pg==='history') loadHistory();
    if (pg==='stats') loadStats();
    if (pg==='vbank') loadVBank();
    if (pg==='projects') loadProjects();
    if (pg==='pron') loadPron();
  });
});

// ── ENGINE SELECTOR ───────────────────────────────────
function setEng(el, name) {
  curEng = name;
  document.querySelectorAll('.eng-tile').forEach(t => {
    t.className = 'eng-tile';
  });
  const map = {kokoro:'active-kok',f5:'active-f5',chatterbox:'active-cha',edge:'active-edg'};
  el.classList.add(map[name]||'');
  document.querySelectorAll('.ep').forEach(p => p.classList.remove('act'));
  document.getElementById('ep-'+name).classList.add('act');
}

// ── VOICE GRID ────────────────────────────────────────
function buildKVG() {
  const vg = document.getElementById('kvg');
  Object.entries(KV).forEach(([k, v]) => {
    const d = document.createElement('div');
    d.className = 'vc' + (k===curKV?' sel':'');
    d.innerHTML = `<div class="va ${v.gender}">${v.label[0]}</div>
      <div class="vn">${v.label}</div>
      <div class="vd">${v.desc.substring(0,22)}</div>`;
    d.onclick = () => {
      document.querySelectorAll('.vc').forEach(x=>x.classList.remove('sel'));
      d.classList.add('sel'); curKV = k;
    };
    vg.appendChild(d);
  });
}

// ── EDGE SELECTS ──────────────────────────────────────
function buildEdge() {
  const vs = document.getElementById('edge-v');
  Object.entries(EV).forEach(([k,v]) => {
    const o = document.createElement('option');
    o.value = k; o.textContent = `${v.label} (${v.lang})`;
    vs.appendChild(o);
  });
  const rs = document.getElementById('edge-r');
  ER.forEach(r => { const o=document.createElement('option'); o.value=r; o.textContent=r; rs.appendChild(o); });
  rs.value = '+0%';
  const ps = document.getElementById('edge-p');
  EP.forEach(p => { const o=document.createElement('option'); o.value=p; o.textContent=p; ps.appendChild(o); });
  ps.value = '+0Hz';
}

// ── XTTS SPEAKERS ─────────────────────────────────────
async function buildXTTS() {
  const sel = document.getElementById('xt-speaker');
  if (!sel) return;
  try {
    const sp = await fetch('/api/xtts/speakers').then(r=>r.json());
    sp.forEach(s => { const o=document.createElement('option'); o.value=s; o.textContent=s; sel.appendChild(o); });
  } catch {}
}

// ── EQ SELECT ─────────────────────────────────────────
function buildEQ() {
  const s = document.getElementById('g-eq');
  EQ.forEach(e => { const o=document.createElement('option'); o.value=e; o.textContent=e; s.appendChild(o); });
  // batch voice
  const bv = document.getElementById('bt-v');
  Object.entries(KV).forEach(([k,v])=>{ const o=document.createElement('option'); o.value=k; o.textContent=v.label; bv.appendChild(o); });
}

// ── TOGGLES ───────────────────────────────────────────
function tog(el, name) {
  el.classList.toggle('on');
  toggs[name] = el.classList.contains('on');
}

// ── TEXT INPUT ────────────────────────────────────────
function onTxt(el) {
  const c = el.value.length;
  const w = el.value.trim().split(/\\s+/).filter(Boolean).length;
  document.getElementById('g-cnt').textContent = c;
  document.getElementById('g-words').textContent = w ? w+' palavras' : '';
  localStorage.setItem('qwn3_draft', el.value);
}

// ── FILE UPLOAD ───────────────────────────────────────
function onUp(inp, zoneId, lblId) {
  const f = inp.files[0];
  if (!f) return;
  document.getElementById(zoneId).classList.add('has');
  document.getElementById(lblId).textContent = '✓ ' + f.name;
}

// ── STATUS ────────────────────────────────────────────
function setStatus(id, type, msg) {
  const el = document.getElementById(id);
  el.className = 'status show ' + type;
  el.innerHTML = type==='load'
    ? `<div class="spin"></div>${msg}`
    : `<span style="font-size:15px">${type==='ok'?'✓':'✕'}</span>${msg}`;
}
function clrStatus(id) {
  document.getElementById(id).className = 'status';
}

// ── TOAST ─────────────────────────────────────────────
function toast(msg, icon='✓', dur=3000) {
  const t = document.createElement('div');
  t.className = 'toast';
  t.innerHTML = `<span style="font-size:16px">${icon}</span>${msg}`;
  document.getElementById('toasts').appendChild(t);
  setTimeout(() => {
    t.classList.add('out');
    setTimeout(() => t.remove(), 300);
  }, dur);
}

// ── WAVEFORM CANVAS ───────────────────────────────────
function initWV() {
  const c = document.getElementById('wv');
  const dpr = window.devicePixelRatio || 1;
  c.width  = c.offsetWidth * dpr;
  c.height = c.offsetHeight * dpr;
  wvCtx = c.getContext('2d');
  wvCtx.scale(dpr, dpr);
  return wvCtx;
}

function drawWVBars(ctx, data, loading) {
  const W = ctx.canvas.offsetWidth || ctx.canvas.width;
  const H = ctx.canvas.offsetHeight || ctx.canvas.height;
  ctx.clearRect(0, 0, W, H);
  const n = data.length || 80;
  const bw = W / n * 0.55;
  const gap = W / n * 0.45;
  const t = Date.now();
  for (let i = 0; i < n; i++) {
    let h;
    if (loading) {
      h = (Math.sin(t * 0.003 + i * 0.35) * 0.4 + 0.6) * H * 0.75;
    } else {
      h = Math.max((data[i] || 0) * H * 0.92, 2);
    }
    const x = i * (bw + gap);
    const y = (H - h) / 2;
    const grad = ctx.createLinearGradient(0, y, 0, y+h);
    grad.addColorStop(0, '#818cf8');
    grad.addColorStop(1, '#22d3ee');
    ctx.fillStyle = grad;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, Math.max(bw,1.5), h, 2);
    else ctx.rect(x, y, Math.max(bw,1.5), h);
    ctx.fill();
  }
}

function animWV(data, loading) {
  if (!wvCtx) initWV();
  if (wvRaf) cancelAnimationFrame(wvRaf);
  if (loading) {
    const loop = () => { drawWVBars(wvCtx, [], true); wvRaf = requestAnimationFrame(loop); };
    loop();
  } else {
    if (wvRaf) cancelAnimationFrame(wvRaf);
    drawWVBars(wvCtx, data, false);
  }
}

// ── GENERATE ─────────────────────────────────────────
async function doGen() {
  const txt = document.getElementById('g-txt').value.trim();
  if (!txt) { toast('Escreva um texto primeiro', '✕'); return; }
  const btn = document.getElementById('g-btn');
  btn.disabled = true;
  setStatus('g-status','load','Gerando com '+curEng+'...');
  document.getElementById('g-player').classList.remove('show');

  let params = {};
  if (curEng==='kokoro') {
    params = { voice: curKV, speed: parseFloat(document.getElementById('k-spd').value) };
  } else if (curEng==='edge') {
    params = {
      voice: document.getElementById('edge-v').value,
      rate:  document.getElementById('edge-r').value,
      pitch: document.getElementById('edge-p').value
    };
  } else if (curEng==='f5') {
    const f = document.getElementById('f5-ref').files[0];
    if (!f) { btn.disabled=false; toast('Selecione um áudio de referência','✕'); clrStatus('g-status'); return; }
    return doCloneFrom('f5', txt, f, document.getElementById('f5-rt').value);
  } else if (curEng==='chatterbox') {
    const f = document.getElementById('cb-ref').files[0];
    return doCloneFrom('chatterbox', txt, f, '', parseFloat(document.getElementById('cb-ex').value));
  } else if (curEng==='xtts') {
    const f = document.getElementById('xt-ref').files[0];
    if (f) {
      return doCloneXTTS(txt, f, document.getElementById('xt-lang').value);
    }
    params = {
      language: document.getElementById('xt-lang').value,
      speaker:  document.getElementById('xt-speaker').value
    };
  }

  const presetName = document.getElementById('g-preset').value;
  const opts = Object.assign({}, toggs, {
    pitch: parseFloat(document.getElementById('g-pitch').value),
    eq: document.getElementById('g-eq').value,
    preset: presetName
  });

  // show waveform loader
  const pc = document.getElementById('g-player');
  pc.classList.add('show');
  if (!wvCtx) initWV(); else { if(wvRaf) cancelAnimationFrame(wvRaf); }
  animWV([], true);

  try {
    const res = await fetch('/api/generate', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({engine:curEng, text:txt, params, opts})
    });
    const d = await res.json();
    if (d.error) throw new Error(d.error);
    renderResult(d);
    setStatus('g-status','ok',`✓ ${d.stats.duration_s}s gerado em ${d.duration_s}s · ${d.engine} · ${d.voice}`);
    toast('Áudio gerado com sucesso', '🎵');
  } catch(e) {
    setStatus('g-status','err', e.message);
    animWV([], false);
    toast('Erro: '+e.message, '✕', 5000);
  }
  btn.disabled = false;
}

// ── PRESET + TRANSCRIBE + XTTS HELPERS ───────────────
function applyPreset(name) {
  if (!name) return;
  toast('Preset: '+name, '🎚');
  // visual feedback — could toggle UI toggles to match preset, but backend handles actual application
}

function updateExagLabel(v) {
  const el = document.getElementById('cb-ex-desc');
  if (!el) return;
  const f = parseFloat(v);
  let txt = '';
  if (f < 0.3) txt = f.toFixed(2)+' → monótono / calmo';
  else if (f < 0.6) txt = f.toFixed(2)+' → natural (recomendado)';
  else if (f < 0.85) txt = f.toFixed(2)+' → expressivo';
  else txt = f.toFixed(2)+' → muito dramático';
  el.textContent = txt;
}

async function autoTranscribe(inputId, targetId) {
  const f = document.getElementById(inputId).files[0];
  if (!f) return;
  const spin = document.getElementById('xt-tr-spin');
  if (spin) spin.innerHTML = '<span class="spin" style="display:inline-block;width:10px;height:10px;vertical-align:middle"></span> transcrevendo...';
  const fd = new FormData();
  fd.append('audio', f);
  try {
    const d = await (await fetch('/api/transcribe', {method:'POST', body:fd})).json();
    if (d.ok && d.text) {
      document.getElementById(targetId).value = d.text;
      toast('Transcrito por Whisper','🧠');
    }
  } catch(e) { /* silent */ }
  if (spin) spin.innerHTML = '';
}

async function doCloneXTTS(text, file, language) {
  const fd = new FormData();
  fd.append('text', text);
  fd.append('engine', 'xtts');
  fd.append('ref_text', document.getElementById('xt-rt').value || '');
  fd.append('audio_ref', file);
  const pc = document.getElementById('g-player');
  pc.classList.add('show');
  if (!wvCtx) initWV();
  animWV([], true);
  try {
    const res = await fetch('/api/generate/clone', {method:'POST', body:fd});
    const d = await res.json();
    if (d.error) throw new Error(d.error);
    renderResult(d);
    setStatus('g-status','ok','✓ XTTS v2 clonado · '+d.stats.duration_s+'s');
    toast('Voz clonada via XTTS v2!', '⭐');
  } catch(e) {
    setStatus('g-status','err', e.message);
    toast('Erro: '+e.message, '✕', 5000);
  }
  document.getElementById('g-btn').disabled = false;
}

async function doCloneFrom(eng, text, file, rtext, exag=0.5) {
  const fd = new FormData();
  fd.append('text', text);
  fd.append('engine', eng);
  fd.append('ref_text', rtext||'');
  fd.append('exaggeration', exag);
  if (file) fd.append('audio_ref', file);

  const pc = document.getElementById('g-player');
  pc.classList.add('show');
  if (!wvCtx) initWV();
  animWV([], true);

  try {
    const res = await fetch('/api/generate/clone', {method:'POST', body:fd});
    const d = await res.json();
    if (d.error) throw new Error(d.error);
    renderResult(d);
    setStatus('g-status','ok','✓ Voz clonada com sucesso');
    toast('Voz clonada!', '🔬');
  } catch(e) {
    setStatus('g-status','err', e.message);
    toast('Erro: '+e.message, '✕', 5000);
  }
  document.getElementById('g-btn').disabled = false;
}

function renderResult(d) {
  lastRes = d;
  const b64 = 'data:audio/wav;base64,' + d.audio_b64;
  const aud = document.getElementById('g-audio');
  aud.src = b64;

  // waveform
  animWV(d.waveform || [], false);

  // meta
  const s = d.stats || {};
  const metas = [
    {v: s.duration_s+'s', l:'Duração'},
    {v: s.peak_db+'dB', l:'Peak'},
    {v: (s.sample_rate/1000)+'kHz', l:'Sample Rate'},
    {v: d.engine, l:'Engine'},
  ];
  document.getElementById('g-ameta').innerHTML = metas.map(m =>
    `<div class="am"><div class="am-v">${m.v}</div><div class="am-l">${m.l}</div></div>`
  ).join('');

  // download links
  const dlw = document.getElementById('g-dl');
  dlw.href = b64; dlw.download = d.filename || 'audio.wav';

  if (d.exports && d.exports.mp3) {
    const dlm = document.getElementById('g-dl-mp3');
    dlm.href = '/api/audio/'+d.exports.mp3.split(/[\\/]/).pop();
    dlm.style.display = '';
  }
  if (d.exports && d.exports.srt) {
    const dls = document.getElementById('g-dl-srt');
    dls.href = '/api/audio/'+d.exports.srt.split(/[\\/]/).pop();
    dls.style.display = '';
  }
}

function showCost() {
  if (!lastRes) return;
  const c = lastRes.cost || {};
  const el = document.getElementById('g-cost');
  el.classList.add('show');
  el.innerHTML = `
    <div style="font-size:12px;line-height:2.2;color:var(--t2)">
      ElevenLabs cobraria: <b style="color:var(--ros)">$${c.elevenlabs_usd||0}</b>
      por ${c.chars||0} chars<br>
      QWN3-TTS: <b style="color:var(--em)">$0.00</b> — economia de
      <b style="color:var(--ind)">100%</b>
    </div>`;
}

// ── CLONE PAGE ───────────────────────────────────────
async function doClone() {
  const file = document.getElementById('cl-ref').files[0];
  if (!file) { toast('Selecione um áudio de referência','✕'); return; }
  const txt = document.getElementById('cl-txt').value.trim();
  if (!txt) { toast('Escreva o texto','✕'); return; }
  const eng = document.getElementById('cl-eng').value;
  document.getElementById('cl-btn').disabled = true;
  setStatus('cl-status','load','Clonando voz...');
  const fd = new FormData();
  fd.append('text',txt); fd.append('engine',eng);
  fd.append('ref_text', document.getElementById('cl-rt').value);
  fd.append('audio_ref', file);
  try {
    const d = await (await fetch('/api/generate/clone',{method:'POST',body:fd})).json();
    if (d.error) throw new Error(d.error);
    const b = 'data:audio/wav;base64,'+d.audio_b64;
    document.getElementById('cl-audio').src = b;
    document.getElementById('cl-dl').href = b;
    document.getElementById('cl-dl').download = d.filename||'clone.wav';
    document.getElementById('cl-player').classList.add('show');
    setStatus('cl-status','ok','Voz clonada com sucesso!');
    toast('Clonagem concluída!','🔬');
  } catch(e) {
    setStatus('cl-status','err', e.message);
    toast('Erro: '+e.message,'✕',5000);
  }
  document.getElementById('cl-btn').disabled = false;
}

// ── BATCH ─────────────────────────────────────────────
async function doBatch() {
  const lines = document.getElementById('bt-txt').value.split('\\n').filter(l=>l.trim());
  if (!lines.length) { toast('Adicione textos','✕'); return; }
  const eng = document.getElementById('bt-eng').value;
  const params = eng==='kokoro' ? {voice: document.getElementById('bt-v').value} : {};
  document.getElementById('bt-btn').disabled = true;
  setStatus('bt-status','load','Gerando '+lines.length+' áudios...');
  document.getElementById('bt-results').innerHTML = '';
  try {
    const d = await (await fetch('/api/batch',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({engine:eng, texts:lines, params, opts:{}})
    })).json();
    const div = document.getElementById('bt-results');
    d.results.forEach(r => {
      div.innerHTML += `<div class="bt-row">
        <span class="${r.ok?'bt-ok':'bt-err'}">${r.ok?'✓':'✕'}</span>
        <span style="flex:1;font-size:11px;color:var(--t2)">${r.text}</span>
        ${r.ok ? `<span style="font-size:10px;color:var(--t3)">${r.audio_s}s</span>
          <a class="btn-sm" href="/api/audio/${r.filename}" download>⬇</a>` : ''}
      </div>`;
    });
    setStatus('bt-status','ok',`${d.results.filter(r=>r.ok).length}/${d.total} gerados`);
  } catch(e) {
    setStatus('bt-status','err', e.message);
  }
  document.getElementById('bt-btn').disabled = false;
}

// ── PODCAST ───────────────────────────────────────────
function addPodLine() {
  const id = Date.now();
  const div = document.createElement('div');
  div.id = 'pl-'+id;
  div.style.cssText = 'display:flex;gap:8px;margin-bottom:8px;align-items:flex-start';
  const voiceOpts = Object.entries(KV).map(([k,v])=>`<option value="${k}">${v.label}</option>`).join('');
  div.innerHTML = `
    <select style="width:130px;flex-shrink:0">${voiceOpts}</select>
    <textarea rows="2" placeholder="Fala do locutor..." style="flex:1;min-height:52px"></textarea>
    <button class="btn-sm danger" onclick="document.getElementById('pl-${id}').remove()">✕</button>`;
  document.getElementById('pod-lines').appendChild(div);
}

async function doPodcast() {
  const lines = document.getElementById('pod-lines').querySelectorAll('div[id^="pl-"]');
  const script = [...lines].map(l => ({
    voice: l.querySelector('select').value,
    text:  l.querySelector('textarea').value.trim(),
    engine:'kokoro'
  })).filter(s=>s.text);
  if (!script.length) { toast('Adicione linhas ao script','✕'); return; }
  setStatus('pod-status','load','Gerando podcast...');
  try {
    const d = await (await fetch('/api/podcast',{
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({script})
    })).json();
    if (d.error) throw new Error(d.error);
    const b = 'data:audio/wav;base64,'+d.audio_b64;
    document.getElementById('pod-audio').src = b;
    document.getElementById('pod-dl').href = b;
    document.getElementById('pod-dl').download = d.filename||'podcast.wav';
    document.getElementById('pod-player').classList.add('show');
    setStatus('pod-status','ok','Podcast gerado: '+d.duration_s+'s');
    toast('Podcast pronto!','🎙');
  } catch(e) { setStatus('pod-status','err',e.message); }
}

// ── VOICE BANK ────────────────────────────────────────
async function saveToBank() {
  const name = prompt('Nome para esta configuração de voz:');
  if (!name) return;
  const params = curEng==='kokoro'
    ? {voice:curKV, speed:parseFloat(document.getElementById('k-spd').value)}
    : curEng==='edge'
    ? {voice:document.getElementById('edge-v').value, rate:document.getElementById('edge-r').value, pitch:document.getElementById('edge-p').value}
    : {};
  await fetch('/api/voices/saved',{
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({name, engine:curEng, params, description:name})
  });
  toast('Voz salva!','🗄️');
}

async function saveVoice() {
  const name = document.getElementById('vb-name').value.trim();
  if (!name) { toast('Escreva um nome','✕'); return; }
  const params = curEng==='kokoro'
    ? {voice:curKV, speed:parseFloat(document.getElementById('k-spd').value)}
    : {};
  await fetch('/api/voices/saved',{
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({name, engine:curEng, params, description:document.getElementById('vb-desc').value})
  });
  toast('Voz salva!','🗄️'); loadVBank();
}

async function loadVBank() {
  const data = await fetch('/api/voices/saved').then(r=>r.json()).catch(()=>[]);
  const div = document.getElementById('vb-list');
  div.innerHTML = '';
  if (!data.length) { div.innerHTML='<div style="color:var(--t3);font-size:12px">Nenhuma voz salva.</div>'; return; }
  data.forEach(v => {
    const el = document.createElement('div');
    el.className = 'hi';
    el.innerHTML = `<span class="hbdg ${v.engine}">${v.engine}</span>
      <span class="htxt">${v.name}</span>
      <span style="font-size:10px;color:var(--t3)">${v.used}x</span>
      <button class="btn-sm danger" onclick="delVoice('${v.id}')">✕</button>`;
    div.appendChild(el);
  });
}

async function delVoice(id) {
  await fetch('/api/voices/saved/'+id, {method:'DELETE'});
  loadVBank(); toast('Voz removida','🗑️');
}

// ── HISTORY ───────────────────────────────────────────
async function loadHistory() {
  const q = document.getElementById('h-q')?.value||'';
  const data = await fetch('/api/history?q='+encodeURIComponent(q)).then(r=>r.json()).catch(()=>[]);
  const div = document.getElementById('h-list');
  div.innerHTML = '';
  if (!data.length) { div.innerHTML='<div style="color:var(--t3);font-size:12px">Nenhum histórico.</div>'; return; }
  data.forEach(h => {
    const el = document.createElement('div');
    el.className='hi';
    el.innerHTML = `<span class="hbdg ${h.engine||''}">${h.engine||'?'}</span>
      <span class="htxt">${h.text||''}</span>
      <span style="font-size:10px;color:var(--t3)">${(h.audio_s||0).toFixed?h.audio_s.toFixed(1)+'s':''}</span>
      ${h.path?`<a class="btn-sm" href="/api/audio/${h.path.split(/[\\/]/).pop()}" download>⬇</a>`:''}
      <button class="btn-sm danger" onclick="delHist('${h.id}')">✕</button>`;
    div.appendChild(el);
  });
}

async function delHist(id) {
  await fetch('/api/history/'+id,{method:'DELETE'}); loadHistory();
}

// ── PROJECTS ──────────────────────────────────────────
async function saveProject() {
  const name = document.getElementById('pj-name').value.trim();
  if (!name) { toast('Escreva um nome','✕'); return; }
  const data = { engine:curEng, voice:curKV, text:document.getElementById('g-txt').value };
  await fetch('/api/projects',{
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({name, data})
  });
  toast('Projeto salvo!','📁'); loadProjects();
}

async function loadProjects() {
  const names = await fetch('/api/projects').then(r=>r.json()).catch(()=>[]);
  const div = document.getElementById('pj-list');
  div.innerHTML = '';
  if (!names.length) { div.innerHTML='<div style="color:var(--t3);font-size:12px">Nenhum projeto.</div>'; return; }
  names.forEach(name => {
    const el=document.createElement('div');
    el.className='hi';
    el.innerHTML = `<span class="htxt">📁 ${name}</span>
      <button class="btn-sm" onclick="loadProject('${name}')">Abrir</button>
      <button class="btn-sm danger" onclick="delProject('${name}')">✕</button>`;
    div.appendChild(el);
  });
}

async function loadProject(name) {
  const d = await fetch('/api/projects/'+name).then(r=>r.json()).catch(()=>({}));
  if (d.text) document.getElementById('g-txt').value = d.text;
  toast('Projeto carregado: '+name,'📁');
  document.querySelector('[data-page="studio"]').click();
}

async function delProject(name) {
  await fetch('/api/projects/'+name,{method:'DELETE'}); loadProjects();
}

// ── PRONUNCIAÇÃO ──────────────────────────────────────
async function loadPron() {
  const data = await fetch('/api/pronunciations').then(r=>r.json()).catch(()=>({}));
  const div = document.getElementById('pr-list');
  div.innerHTML = '';
  if (!Object.keys(data).length) { div.innerHTML='<div style="color:var(--t3);font-size:12px">Dicionário vazio.</div>'; return; }
  Object.entries(data).forEach(([w,p]) => {
    const el=document.createElement('div');
    el.className='hi';
    el.innerHTML=`<span class="htxt"><b>${w}</b> → ${p}</span>
      <button class="btn-sm danger" onclick="delPron('${w}')">✕</button>`;
    div.appendChild(el);
  });
}

async function addPron() {
  const w=document.getElementById('pr-w').value.trim();
  const p=document.getElementById('pr-p').value.trim();
  if(!w||!p) return;
  await fetch('/api/pronunciations',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({word:w,phonetic:p})});
  document.getElementById('pr-w').value='';
  document.getElementById('pr-p').value='';
  loadPron(); toast('Pronúncia adicionada','📖');
}

async function delPron(w) {
  await fetch('/api/pronunciations/'+encodeURIComponent(w),{method:'DELETE'}); loadPron();
}

// ── STATS ─────────────────────────────────────────────
async function loadStats() {
  const d = await fetch('/api/stats').then(r=>r.json()).catch(()=>({}));
  const cards=[
    {v:d.total_generated||0,l:'Áudios Gerados'},
    {v:d.total_chars||0,l:'Chars Processados'},
    {v:(d.total_seconds||0)+'s',l:'Áudio Total'},
    {v:d.history_count||0,l:'No Histórico'},
    {v:d.projects||0,l:'Projetos'},
    {v:'$0.00',l:'Custo Total'},
  ];
  document.getElementById('st-cards').innerHTML = cards.map(c=>
    `<div class="sc"><div class="sn">${c.v}</div><div class="sl">${c.l}</div></div>`
  ).join('');
  const cost = ((d.total_chars||0)/1000*0.30).toFixed(4);
  document.getElementById('st-cost').innerHTML = `
    <div class="ct">Economia vs ElevenLabs</div>
    <div style="font-size:13px;line-height:2.4;color:var(--t2)">
      ElevenLabs custaria: <b style="color:var(--ros)">$${cost}</b><br>
      QWN3-TTS: <b style="color:var(--em)">$0.00</b><br>
      Economia total: <b style="color:var(--ind);font-size:16px">$${cost} (100%)</b>
    </div>`;
}

// ── ENGINE STATUS POLL ────────────────────────────────
async function pollInfo() {
  try {
    const i = await fetch('/api/info').then(r=>r.json());
    const el = document.getElementById('b-eng');
    if (i.engines_loaded.length) {
      el.textContent = i.engines_loaded.length + ' engines · online';
      el.className = 'hbadge live';
    } else {
      el.textContent = 'carregando engines...';
    }
  } catch {}
}

// ── KEYBOARD SHORTCUTS ────────────────────────────────
document.addEventListener('keydown', e => {
  if ((e.ctrlKey||e.metaKey) && e.key==='Enter') {
    e.preventDefault();
    if (document.getElementById('page-studio').classList.contains('active')) doGen();
  }
});

// ── INIT ──────────────────────────────────────────────
window.onload = () => {
  buildKVG();
  buildEdge();
  buildEQ();
  buildXTTS();
  updateExagLabel(0.5);

  // restore draft
  const draft = localStorage.getItem('qwn3_draft');
  if (draft) {
    const ta = document.getElementById('g-txt');
    ta.value = draft;
    onTxt(ta);
  }

  // poll engine status
  pollInfo();
  setInterval(pollInfo, 8000);

  // add default podcast line
  addPodLine();

  // resize canvas on window resize
  window.addEventListener('resize', () => { wvCtx = null; });
};
</script>
</body>
</html>"""
    return (H
        .replace("__KV__", kv)
        .replace("__EV__", ev)
        .replace("__EQ__", eq)
        .replace("__ER__", er)
        .replace("__EP__", ep)
        .replace("__GPU__", gpu)
    )
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  QWN3-TTS STUDIO v5.0 — Multi-Engine Production")
    print(f"  GPU: {GPU_NAME} {GPU_VRAM}")
    print("  Engines: Kokoro · F5-TTS · Chatterbox · Edge TTS")
    print("  http://localhost:7860")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="warning",
                timeout_keep_alive=300)

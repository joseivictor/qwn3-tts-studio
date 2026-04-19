# QWN3 TTS Studio

> Professional text-to-speech platform with 50+ features. No subscriptions. No limits. Better than ElevenLabs.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Why QWN3 TTS Studio?

| Feature | ElevenLabs | QWN3 TTS Studio |
|---------|-----------|-----------------|
| Price | $5–$330/month | **Free** |
| API access | Paid tier | **Free** |
| Audio processing | Basic | **50+ features** |
| Batch processing | Limited | **Unlimited** |
| Local / offline | No | **Yes** |
| Open source | No | **Yes** |
| Custom pronunciations | No | **Yes** |
| Podcast multi-speaker | No | **Yes** |
| Voice fingerprinting | No | **Yes** |
| SRT subtitle export | No | **Yes** |

---

## 50+ Features

### Generation
- Text-to-speech with multiple providers (Kokoro, Edge TTS, XTTS)
- Voice cloning from audio samples
- SSML support (rate, pitch, volume, breaks)
- Auto-punctuation and text cleaning
- Language detection (auto)

### Audio Processing
- **Normalize** — auto-level to target dBFS
- **Speed control** — 0.5× to 3.0× without pitch change (librosa)
- **Pitch shift** — ±12 semitones
- **Silence trim** — remove leading/trailing silence
- **Reverb** — room size + wet level
- **Echo** — delay + decay
- **Denoise** — spectral subtraction
- **Stereo** — mono to stereo conversion
- **EQ** — 5 presets: Neutro, Quente, Brilhante, Rádio, Podcast, Telefone
- **Dynamic compression** — threshold + ratio
- **Fade in/out** — ms-level control
- **Padding** — add silence before/after

### Export
- WAV (lossless)
- **MP3** (quality 0–9)
- **FLAC** (lossless compressed)
- **SRT subtitles** — synced with audio

### Advanced
- **Batch processing** — process hundreds of texts in one click
- **Podcast mode** — multi-speaker conversations with different voices
- **Voice A/B compare** — side-by-side comparison
- **Background music** — mix TTS with ambient audio
- **Voice aging** — make voice sound younger/older
- **Gender morph** — shift voice gender
- **Pronunciation dictionary** — custom word → phoneme
- **Voice fingerprinting** — identify/tag voice
- **Processing pipeline** — chain effects in sequence
- **Audio join** — concatenate multiple files
- **Audio trim** — cut start/end by ms
- **Waveform data** — export for visualization

### Project Management
- **Project save/load** — JSON snapshot of all settings
- **History** — full log of every generation (CSV export)
- **Job queue** — async batch processing
- **Cost estimator** — estimate API cost before running
- **Usage stats** — chars generated, requests, uptime
- **API key management** — generate/revoke access keys

### Utility
- **Text chunker** — split long texts at sentence boundaries
- **Text stats** — word count, char count, estimated duration
- **Preview mode** — quick 50-char preview
- **SSML parser** — validate and preview SSML markup
- **Pronunciation editor** — visual dictionary UI

---

## Installation

```bash
git clone https://github.com/joseivictor/qwn3-tts-studio
cd qwn3-tts-studio
pip install -r requirements.txt
```

### Requirements

```
fastapi
uvicorn
kokoro-tts      # or: pip install kokoro
edge-tts
librosa
soundfile
pydub
numpy
```

---

## Usage

```bash
python tts_studio.py
```

Open **http://localhost:7860** in your browser.

---

## API

All features are available via REST API:

```bash
# Generate speech
curl -X POST http://localhost:7860/api/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "af_heart", "speed": 1.0}'

# Batch generation
curl -X POST http://localhost:7860/api/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Line 1", "Line 2"], "voice": "af_heart"}'

# Podcast multi-speaker
curl -X POST http://localhost:7860/api/podcast \
  -H "Content-Type: application/json" \
  -d '{"segments": [{"text": "Hello", "speaker": "HOST"}, {"text": "Hi!", "speaker": "GUEST"}]}'
```

Full API docs at **http://localhost:7860/docs** (Swagger UI).

---

## Web Interface

11 pages accessible from the sidebar:

1. **Gerar** — main TTS generation
2. **Design de Voz** — EQ, speed, pitch, effects
3. **Clonagem** — voice cloning
4. **Batch** — bulk processing
5. **Podcast** — multi-speaker audio
6. **Comparar** — A/B voice comparison
7. **Histórico** — generation history
8. **Projetos** — save/load projects
9. **Pronúncia** — pronunciation dictionary
10. **Config** — settings and API keys
11. **Stats** — usage statistics

---

## TTS Providers

| Provider | Quality | Speed | Cost |
|----------|---------|-------|------|
| Kokoro | Excellent | Fast | Free/Local |
| Edge TTS | Good | Fast | Free (Microsoft) |
| XTTS v2 | Excellent (cloning) | Medium | Free/Local |

---

## License

MIT — free to use commercially, modify, and distribute.

---

*Built with FastAPI, Kokoro TTS, librosa, pydub, and Python.*

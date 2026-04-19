@echo off
title QWN3-TTS Studio v5.0
cd /d "%~dp0"
echo.
echo  ============================================
echo   QWN3-TTS Studio v5.0 - Production Edition
echo   Kokoro / F5-TTS / Chatterbox / Edge TTS
echo   http://localhost:7860
echo  ============================================
echo.
tts_env\Scripts\python.exe -X utf8 tts_studio.py
pause

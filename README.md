# transcription-dev

An automated speech-to-text tool that transcribes audio files linked to ELAN files (`.eaf`) and writes the generated text directly onto the ELAN timeline.

## Overview
- Automatically transcribes audio into text.
- Synchronizes and writes output directly onto the corresponding ELAN timeline (tier).

## Environment & Requirements
- **Target Environment:** Designed specifically for **Google Colaboratory (Google Colab)**.
- **Audio Format:** Supports **WAV files only** (`.wav`).
- **Dependencies:** No manual or separate library installations are required prior to execution.

## Usage
1. Open the notebook in Google Colab.
2. Ensure your `.eaf` file and the corresponding `.wav` audio file are placed in the specified directory.
3. Run `Transcript.py` to start automatic transcription and sync text to the ELAN timeline.

# Transcribe-Align-Diarize

```bash
uv run main.py --audio-file ./hello-world.mp3 --diarize
```

```json
{
  "words": [
    {
      "speaker": 0,
      "start": 0.32,
      "end": 0.74,
      "word": "hello",
      "confidence": 0.98
    },
    {
      "speaker": 1,
      "start": 0.86,
      "end": 1.06,
      "word": "world",
      "confidence": 0.95
    }
  ],
  "detected_language": "en"
}
```

A toolkit for audio processing that performs speech transcription, forced alignment, and speaker diarization using state-of-the-art deep learning models (faster whisper, NVIDIA NeMo diarization, and ctc forced alignment using MMS-300m).

Run via cli, as a local api, or deploy to [beam.cloud](https://beam.cloud)

Based on [MahmoudAshraf97/whisper-diarization](https://github.com/MahmoudAshraf97/whisper-diarization)

## Features

- **Speech Transcription**: Convert speech to text using Whisper models
- **Word-Level Alignment**: Precise timestamps for each word in the transcript
- **Speaker Diarization**: Identify and label different speakers in audio recordings
- **Multiple Model Options**:
  - `large-v3`: High accuracy Whisper model
  - `distil-large-v3`: Faster, distilled version
  - Any model that runs with FasterWhisper
    - https://huggingface.co/models?other=faster-whisper

## Requirements

- Python 3.10+
- CUDA-capable GPU (CUDA 12.8 recommended)
- [uv](https://github.com/astral-sh/uv) (Recommended for python package management)
- System packages:
  - FFmpeg
  - libsox-dev
  - CUDA 12.x
  - cuDNN 9 for CUDA 12

### System Package Installation

#### Linux (Debian/Ubuntu)
```bash
# Install FFmpeg and libsox-dev
sudo apt-get update
sudo apt-get install -y ffmpeg libsox-dev
```

#### macOS
```bash
# Install FFmpeg and libsox-dev through Homebrew
brew update
brew install ffmpeg sox
```

> Note: For CUDA 12.x and cuDNN 9 installation, please follow NVIDIA's official documentation for your specific operating system.

## Installation

```bash
# Clone the repository
git clone https://github.com/tylerkahn/transcribe-align-diarize.git
cd transcribe-align-diarize

uv venv .venv
source .venv/bin/activate
uv sync
```

## Usage

### Command-Line Interface (CLI)

#### Basic Transcription

```bash
uv run main.py --audio-file ./apples.mp3
```

#### Transcription with Speaker Diarization

```bash
uv run main.py --audio-file ./apples.mp3 --diarize
```

#### Forced Alignment with Existing Transcript

```bash
uv run main.py --audio-file ./apples.mp3 --transcript "I'm going to the store because I need apples"
```

#### CLI Arguments

- `--audio-file`: Path to the audio file
- `--url`: Alternative to audio-file, URL to audio file
- `--language`: Language code (e.g., "en", "fr") or "auto" for detection
- `--diarize`: Flag to enable speaker diarization
- `--transcript`: Text to align with audio (for forced alignment)
- `--transcript-file`: File containing text to align with audio
- `--whisper-model-name`: Model to use (large-v3, distil-large-v3, etc.)
- `--model-cache-dir`: Directory to cache models (default: ./.model-cache)
- `--output`: Output file path for results (default: prints to stdout)

### RESTful API

The toolkit also provides a RESTful API through `api.py`:

```bash
uv run api.py --whisper-model-name large-v3 --port 8000
```

This starts a local server on port 8000 with the following endpoints:

#### Endpoints

- `/transcribe`: Transcribe and align audio
- `/force-align`: Force align existing transcript with audio

#### API Examples

**Transcription Request:**

```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/audio.mp3",
    "language": "en",
    "diarize": true
  }'
```

**Force Alignment Request:**

```bash
curl -X POST "http://localhost:8000/force-align" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_base64": "UklGRgZrAgBXQVZFZm10IBAAA...",
    "diarize": true,
    "transcript": "This is the text that needs to be aligned with the audio."
  }'
```

## Response Format

### Transcription (With Diarization)

```json
{
  "words": [
    {
      "speaker": 0,
      "start": 0.32,
      "end": 0.74,
      "word": "hello",
      "confidence": 0.98
    },
    {
      "speaker": 1,
      "start": 0.86,
      "end": 1.06,
      "word": "world",
      "confidence": 0.95
    }
  ],
  "detected_language": "en"
}
```

### Forced Alignment (With Diarization)

```json
[
    {
      "speaker": 0,
      "start": 0.32,
      "end": 0.74,
      "word": "hello",
      "confidence": 0.98
    },
    {
      "speaker": 1,
      "start": 0.86,
      "end": 1.06,
      "word": "world",
      "confidence": 0.95
    }
],
```

## Deployment

### Beam Deployment

This toolkit can be deployed on [Beam](https://beam.cloud/):

```bash
# Install Beam client using uv
uv pip install beam-client

# Deploy to Beam
uv run beam deploy beam_predict.py:predict_force_align
uv run beam deploy beam_predict.py:predict_transcribe_large_v3
```

## Architecture

- `transcribe.py`: Handles transcription using Whisper models
- `force_align.py`: Provides forced alignment functionality
- `main.py`: Command-line interface
- `api.py`: RESTful API interface

## License

MIT License

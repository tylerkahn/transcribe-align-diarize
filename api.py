import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import argparse
from contextlib import asynccontextmanager

from transcribe import predict as predict_transcribe, load_models
from force_align import predict as predict_force_align

# Global variables to store loaded models
pipeline = None
alignment_model = None
alignment_tokenizer = None
whisper_model = None

class TranscribeRequest(BaseModel):
    language: str = "auto"
    audio_base64: Optional[str] = None
    url: Optional[str] = None
    diarize: bool = False

class ForceAlignRequest(BaseModel):
    language: str = "auto"
    audio_base64: Optional[str] = None
    url: Optional[str] = None
    transcript: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models during startup
    global pipeline, alignment_model, alignment_tokenizer, whisper_model
    print(f"Loading models: {app.state.whisper_model_name}")
    pipeline, alignment_model, alignment_tokenizer, whisper_model = load_models(
        app.state.whisper_model_name,
        app.state.model_cache_dir
    )
    yield
    # Cleanup (if needed)
    print("Shutting down API server")

app = FastAPI(lifespan=lifespan)

@app.post("/transcribe")
async def transcribe_endpoint(request: TranscribeRequest):
    try:
        if not request.audio_base64 and not request.url:
            raise HTTPException(
                status_code=400,
                detail="Either audio_base64 or url must be provided"
            )

        result = predict_transcribe(
            language=request.language,
            audio_base64=request.audio_base64,
            url=request.url,
            diarize=request.diarize,
            whisper_model_name=app.state.whisper_model_name,
            model_cache_dir=app.state.model_cache_dir,
            whisper_batch_size=app.state.whisper_batch_size,
            alignment_batch_size=app.state.alignment_batch_size,
            skip_alignment=True,
            # Pass the pre-loaded models as function
            load_models_func=lambda *args, **kwargs: (
                pipeline, alignment_model, alignment_tokenizer, whisper_model
            )
        )

        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/force-align")
async def force_align_endpoint(request: ForceAlignRequest):
    try:
        if not request.audio_base64 and not request.url:
            raise HTTPException(
                status_code=400,
                detail="Either audio_base64 or url must be provided"
            )

        if not request.transcript:
            raise HTTPException(
                status_code=400,
                detail="Transcript must be provided for force alignment"
            )

        result = predict_force_align(
            language=request.language,
            audio_base64=request.audio_base64,
            url=request.url,
            transcript=request.transcript,
            whisper_model_name=app.state.whisper_model_name,
            model_cache_dir=app.state.model_cache_dir,
            alignment_batch_size=app.state.alignment_batch_size,
            # Pass the pre-loaded models as function
            load_models_func=lambda *args, **kwargs: (
                pipeline, alignment_model, alignment_tokenizer
            )
        )

        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    parser = argparse.ArgumentParser(description="Start the transcription API server")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to run the server on")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to run the server on")
    parser.add_argument("--whisper-model-name", type=str, default="large-v3",
                        help="Whisper model name to use")
    parser.add_argument("--model-cache-dir", type=str, default="./.model-cache",
                        help="Directory to cache models")
    parser.add_argument("--whisper-batch-size", type=int, default=8,
                        help="Batch size for Whisper inference (default: 8)")
    parser.add_argument("--alignment-batch-size", type=int, default=4,
                        help="Batch size for alignment (default: 4)")

    args = parser.parse_args()

    # Store model parameters in app state
    app.state.whisper_model_name = args.whisper_model_name
    app.state.model_cache_dir = args.model_cache_dir
    app.state.whisper_batch_size = args.whisper_batch_size
    app.state.alignment_batch_size = args.alignment_batch_size

    # Print API documentation
    print_api_documentation(args.host, args.port)

    # Run the server
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

def print_api_documentation(host, port):
    """
    Print API usage documentation when the server starts
    """
    server_url = f"http://{host}:{port}" if host != "0.0.0.0" else f"http://localhost:{port}"

    print(f"\n Transcription API Server is running at {server_url}")
    print(f"\n POST {server_url}/transcribe - Transcribe audio with optional diarization")
    print(f"\n POST {server_url}/force-align - Force align audio with a transcript")

    print("\nEXAMPLE CURL REQUESTS:")
    print(f"""  curl -X POST {server_url}/transcribe \\
    -H "Content-Type: application/json" \\
    -d '{{
      "url": "https://example.com/audio.mp3",
      "diarize": true
    }}'
    """)

    print(f"""  curl -X POST {server_url}/force-align \\
    -H "Content-Type: application/json" \\
    -d '{{
      "url": "https://example.com/audio.mp3",
      "transcript": "This is the text that needs to be aligned with the audio."
    }}'
    """)

    print("Documentation UI available at:")
    print(f"  {server_url}/docs")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

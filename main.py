import argparse
import base64
import json
import sys
import transcribe
import force_align

def main():
    parser = argparse.ArgumentParser(description="Audio transcription, alignment and diarization tool")

    # Required arguments group (either audio_file or url must be provided)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--audio-file", type=str, help="Path to audio file to transcribe")
    input_group.add_argument("--url", type=str, help="URL to audio file to transcribe")

    # Optional arguments that switch behavior
    input_group2 = parser.add_mutually_exclusive_group(required=False)
    input_group2.add_argument("--transcript", type=str, help="A string e.g. \"hello world\", if provided, transcription with whisper will be bypassed")
    input_group2.add_argument("--transcript-file", type=str, help="File path to plain text, if provided, transcription with whisper will be bypassed")

    # Optional arguments
    parser.add_argument("--language", type=str, default="auto", help="Language code (e.g. 'en', 'he', or 'fr') (default: auto). If auto, the language will be detected automatically.")
    parser.add_argument("--diarize", action="store_true", help="Enable speaker diarization")
    parser.add_argument("--whisper-model-name", type=str, default="large-v3",
                        help="Faster whisper model name (default: large-v3)")
    parser.add_argument("--model-cache-dir", type=str, default="./.model-cache",
                        help="Directory to cache models (default: ./.model-cache)")
    parser.add_argument("--whisper-batch-size", type=int, default=8,
                        help="Batch size for Whisper inference (default: 8)")
    parser.add_argument("--alignment-batch-size", type=int, default=4,
                        help="Batch size for alignment (default: 4)")
    parser.add_argument("--output", type=str, help="Output file path e.g. output.json (default: stdout)")

    args = parser.parse_args()

    # Convert audio file to base64 if provided
    audio_base64 = None
    if args.audio_file:
        try:
            with open(args.audio_file, 'rb') as f:
                audio_bytes = f.read()
                audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        except Exception as e:
            print(f"Error reading audio file: {e}", file=sys.stderr)
            return 1

    # Get transcript from file if provided
    transcript = args.transcript
    if args.transcript_file:
        try:
            with open(args.transcript_file, 'r') as f:
                transcript = f.read()
        except Exception as e:
            print(f"Error reading transcript file: {e}", file=sys.stderr)
            return 1

    # Call appropriate predict function based on input
    if transcript or args.transcript_file:
        # Use force align when transcript is provided
        result = force_align.predict(
            language=args.language,
            audio_base64=audio_base64,
            url=args.url,
            transcript=transcript,
            diarize=args.diarize,
            model_cache_dir=args.model_cache_dir,
            alignment_batch_size=args.alignment_batch_size
        )
    else:
        # Use transcribe when no transcript is provided
        result = transcribe.predict(
            language=args.language,
            audio_base64=audio_base64,
            url=args.url,
            diarize=args.diarize,
            whisper_model_name=args.whisper_model_name,
            model_cache_dir=args.model_cache_dir,
            whisper_batch_size=args.whisper_batch_size,
            alignment_batch_size=args.alignment_batch_size
        )

    # Handle output (file or stdout)
    json_result = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, 'w') as f:
            f.write(json_result)
    else:
        print(json_result)

    return 0


if __name__ == "__main__":
    sys.exit(main())

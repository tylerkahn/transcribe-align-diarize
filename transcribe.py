from whisper_diarization import DiarizationPipeline, helpers, ParallelNemo
from ctc_forced_aligner import load_audio, generate_emissions, preprocess_text, get_alignments, get_spans, postprocess_results

import faster_whisper
from faster_whisper import WhisperModel

import tempfile

from types import SimpleNamespace
from typing import Optional, Callable, Any, Type, Tuple
import numpy as np
import base64
import requests

import torch
import json

from tempfile import NamedTemporaryFile

def load_models(whisper_model_name, model_cache_dir: str | None = None):
    pipeline = DiarizationPipeline(model_cache_dir)
    alignment_model, alignment_tokenizer = pipeline.load_cached_alignment_model("cuda", torch.float16)
    pipeline.download_diarization_model()
    whisper_model = faster_whisper.WhisperModel(
        whisper_model_name,
        device="cuda",
        compute_type="float16",
        download_root=pipeline.WHISPER_CACHE_DIR,
    )
    return (pipeline, alignment_model, alignment_tokenizer, whisper_model)

# load_models_func param is to accomodate beam, see beam_predict.py

def predict(
  language = "auto",
  audio_base64: str | None = None,
  url: Optional[str] = None,
  diarize=False,
  whisper_model_name="large-v3",
  model_cache_dir: str | None = "./.model-cache",
  whisper_batch_size: int = 8,
  alignment_batch_size: int = 4,
  load_models_func: Callable[[str, Optional[str]], Tuple[Type[DiarizationPipeline], Any, Any, Type[WhisperModel]]] = load_models,):

    if not torch.cuda.is_available():
      raise RuntimeError("CUDA is not available. Please check your CUDA installation.")

    (pipeline, alignment_model, alignment_tokenizer, whisper_model) = load_models_func(whisper_model_name, model_cache_dir)
    pipeline.TEMP_DIR = tempfile.mkdtemp()

    # Inputs passed to API
    if language == "auto" or language == "":
      language = None
    iso_lang = helpers.langs_to_iso[language] if language else None

    if diarize == "true" or diarize == True or diarize == "True" or diarize == "1" or diarize == 1:
        diarize = True
    else:
        diarize = False


    if audio_base64 and url:
        return {"error": "Only a base64 audio file OR a URL can be passed to the API."}
    if not audio_base64 and not url:
        return {
            "error": "Please provide either an audio file in base64 string format or a URL."
        }

    binary_data = None

    if audio_base64:
        binary_data = base64.b64decode(audio_base64.encode("utf-8"))
    elif url:
        resp = requests.get(url)
        binary_data = resp.content

    temp = NamedTemporaryFile()
    temp.write(binary_data)
    temp.flush()
    temp.seek(0)
    if diarize:
        nemo = ParallelNemo(temp.name, "cuda", pipeline.TEMP_DIR, pipeline.CACHE_DIR)
        nemo.start()
    temp.seek(0)
    audio_waveform = faster_whisper.decode_audio(temp)
    temp.seek(0)
    waveform = load_audio(temp, dtype=torch.float16, device="cuda")

    whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)


    transcript_segments, info = whisper_pipeline.transcribe(
        audio_waveform,
        language=language,
        batch_size=whisper_batch_size,)

    full_transcript = "".join(segment.text for segment in transcript_segments)

    emissions, stride = generate_emissions(
        alignment_model,
        waveform,
        batch_size=alignment_batch_size,
    )

    tokens_starred, text_starred = preprocess_text(
        full_transcript,
        romanize=True,
        language=helpers.langs_to_iso.get(info.language, None) if language == "auto" else iso_lang,
    )

    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        alignment_tokenizer,
    )

    spans = get_spans(tokens_starred, segments, blank_token)
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)

    if not diarize:
        normalized_words = []
        for word in word_timestamps:
            confidence = float(np.exp(word["score"])) if word["score"] is not None else 1.0
            word_entry = {
                "start": word["start"],
                "end": word["end"],
                "word": word["text"].strip(),
                "confidence": confidence,
            }
            normalized_words.append(word_entry)

        temp.close()
        return {"words": normalized_words, "detected_language": info.language}

    speaker_ts = nemo.wait_for_results()
    temp.close()

    wsm = helpers.get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
    wsm = pipeline._add_punctuation(wsm, SimpleNamespace(language=language), None)

    word_level_output = []
    for word_info, score in zip(wsm, scores):
        confidence = float(np.exp(score)) if score is not None else 1.0
        word_entry = {
            "speaker": word_info["speaker"],
            "start": word_info["start_time"] / 1000.0,
            "end": word_info["end_time"] / 1000.0,
            "word": word_info["word"].strip(),
            "confidence": confidence,
        }
        word_level_output.append(word_entry)

    return {"words": word_level_output, "detected_language": info.language}


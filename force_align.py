from whisper_diarization import DiarizationPipeline, helpers, ParallelNemo
from ctc_forced_aligner import load_audio, generate_emissions, preprocess_text, get_alignments, get_spans, postprocess_results

from typing import Optional, Callable, Any, Type, Tuple

from types import SimpleNamespace
import numpy as np
import base64
import requests

import torch

from tempfile import NamedTemporaryFile, mkdtemp


def load_models(model_cache_dir: str | None = None):
    pipeline = DiarizationPipeline(model_cache_dir)
    alignment_model, alignment_tokenizer = pipeline.load_cached_alignment_model("cuda", torch.float16)
    pipeline.download_diarization_model()
    return (pipeline, alignment_model, alignment_tokenizer)

def predict(
  transcript: str,
  language = "auto",
  audio_base64: str | None = None,
  url: Optional[str] = None,
  diarize=False,
  model_cache_dir: str | None = "./.model-cache",
  alignment_batch_size: int = 4,
  load_models_func: Callable[[str], Tuple[Type[DiarizationPipeline], Any, Any]] = load_models,):
    (pipeline, alignment_model, alignment_tokenizer) = load_models_func(model_cache_dir)

    pipeline.TEMP_DIR = mkdtemp()

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
    waveform = load_audio(temp, dtype=torch.float16, device="cuda")

    emissions, stride = generate_emissions(
        alignment_model,
        waveform,
        batch_size=alignment_batch_size
    )

    tokens_starred, text_starred = preprocess_text(
        transcript,
        romanize=True,
        language=iso_lang,
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


        return normalized_words

    speaker_ts = nemo.wait_for_results()

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
    return word_level_output

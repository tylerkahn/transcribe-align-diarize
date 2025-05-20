from beam import endpoint, Image, Volume, env
import transcribe
import force_align

BEAM_VOLUME_PATH = "./cached_models"

image = Image(
    python_version="python3.10",
    base_image="nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04").add_commands([
        "apt-get install -y ffmpeg libsox-dev build-essential gcc g++ clang"
    ]).add_python_packages([
        'huggingface_hub[hf-transfer]',
        'numpy (>=1.24.3,<2.0.0)',
        'faster-whisper',
        'torchaudio>=2.6.0',
        'torch>=2.7.0',
        'git+https://github.com/tylerkahn/whisper-diarization',
        'requests'
    ]).with_envs("HF_HUB_ENABLE_HF_TRANSFER=1")

def load_models_force_align():
    return force_align.load_models(model_cache_dir=BEAM_VOLUME_PATH),
@endpoint(
    on_start=load_models_force_align,
    name="force-align",
    cpu=2,
    memory="32Gi",
    gpu="A10G",
    image=image,
    keep_warm_seconds=1,
    volumes=[
        Volume(
            name="cached_models",
            mount_path=BEAM_VOLUME_PATH,
        )
    ],
)
def predict_force_align(context, **inputs):
    return force_align.predict(**inputs, load_models_func=lambda a: context.on_start_value)


def load_models_ivrit_large_v3():
    return transcribe.load_models("ivrit-ai/whisper-large-v3-ct2", model_cache_dir=BEAM_VOLUME_PATH)
@endpoint(
    on_start=load_models_ivrit_large_v3,
    name="transcribe-with-whisper-ivrit-large-v3",
    cpu=2,
    memory="32Gi",
    gpu="A10G",
    image=image,
    keep_warm_seconds=1,
    volumes=[
        Volume(
            name="cached_models",
            mount_path=BEAM_VOLUME_PATH,
        )
    ],
)
def predict_transcribe_ivrit_large_v3(context, **inputs):
    return transcribe.predict(**inputs, load_models_func=lambda a,b: context.on_start_value)


def load_large_v3():
    return transcribe.load_models("large-v3", model_cache_dir=BEAM_VOLUME_PATH)
@endpoint(
    on_start=load_large_v3,
    name="transcribe-with-whisper-large-v3",
    cpu=2,
    memory="32Gi",
    gpu="A10G",
    image=image,
    keep_warm_seconds=1,
    volumes=[
        Volume(
            name="cached_models",
            mount_path=BEAM_VOLUME_PATH,
        )
    ],
)
def predict_transcribe_large_v3(context, **inputs):
    return transcribe.predict(**inputs, load_models_func=lambda a,b: context.on_start_value)


def load_distill_large_v3():
    return transcribe.load_models("distil-large-v3", model_cache_dir=BEAM_VOLUME_PATH)
@endpoint(
    on_start=load_distill_large_v3,
    name="transcribe-with-whisper-distill-large-v3",
    cpu=2,
    memory="32Gi",
    gpu="A10G",
    image=image,
    keep_warm_seconds=1,
    volumes=[
        Volume(
            name="cached_models",
            mount_path=BEAM_VOLUME_PATH,
        )
    ],
)
def predict_transcribe_distill_large_v3(context, **inputs):
    return transcribe.predict(**inputs, load_models_func=lambda a,b: context.on_start_value)

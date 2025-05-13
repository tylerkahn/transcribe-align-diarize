from beam import endpoint, Image, Volume, env
import transcribe
import force_align

BEAM_VOLUME_PATH = "./cached_models"

image = Image().from_dockerfile("./Dockerfile")

def load_force_align_models():
    return force_align.load_models(model_cache_dir=BEAM_VOLUME_PATH)
@endpoint(
    on_start=load_force_align_models,
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
    return force_align.predict(**inputs, load_models_func=lambda: context.on_start_value)


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
def predict_ivrit_large_v3(context, **inputs):
    return transcribe.predict(**inputs, load_models_func=lambda: context.on_start_value)


def load_models_large_v3():
    return transcribe.load_models("large-v3", model_cache_dir=BEAM_VOLUME_PATH)
@endpoint(
    on_start=load_models_large_v3,
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
def predict_large_v3(context, **inputs):
    return transcribe.predict(**inputs, load_models_func=lambda: context.on_start_value)


def load_models_distill_large_v3():
    return transcribe.load_models("distil-large-v3", model_cache_dir=BEAM_VOLUME_PATH)
@endpoint(
    on_start=load_models_distill_large_v3,
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
def predict_distill_large_v3(context, **inputs):
    return transcribe.predict(**inputs, load_models_func=lambda: context.on_start_value)

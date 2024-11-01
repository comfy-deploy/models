from comfy_models.base.comfy_utils import Config

config = Config(
    name="sd3_5_large",
    models_to_cache=[
        "checkpoints/sd3.5_large.safetensors",
        "clip/clip_l.safetensors",
        "clip/clip_g.safetensors",
        "clip/t5xxl_fp16.safetensors",
    ],
)

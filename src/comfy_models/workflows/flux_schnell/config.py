from comfy_models.base.comfy_utils import Config

config = Config(
    name="flux_schnell",
    # memroy snspshot
    models_to_cache=[
        "unet/flux1-schnell.sft",
        "clip/clip_l.safetensors",
        "clip/clip_g.safetensors",
        "clip/t5xxl_fp16.safetensors",
        "vae/ae.sft",
    ],
    warmup_workflow=True,
)

from comfy_models.base.comfy_utils import Config

config = Config(
    name="flux_dev",
    # memroy snspshot
    models_to_cache=[
        "unet/flux1-dev.sft",
        "clip/clip_l.safetensors",
        "clip/clip_g.safetensors",
        "clip/t5xxl_fp16.safetensors",
        "vae/ae.sft",
    ],
    # run_twice=True,
    # nodes to preload during comfyui cold start
    # nodes_to_preload=["UNETLoader", "VAELoader", "DualCLIPLoader"],
    warmup_workflow=True,
)
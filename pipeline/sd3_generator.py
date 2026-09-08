"""
SD3 Generator — generates synthetic radiology images with Stable Diffusion 3
running locally on a GPU, in either text-to-image or image-to-image mode.

Unlike the DALL-E and Gemini back-ends, SD3 is a local model rather than an HTTP
API. Two consequences shape this module:

  1. The pipeline is a multi-gigabyte object that takes 30-90s to reach the GPU,
     so it is cached in a module-level singleton and preloaded by the
     orchestrator before the record loop. It is never constructed per image.
  2. Reference images cannot be passed as API attachments. In img2img mode the
     reference X-ray becomes the initial latent; in txt2img mode it is ignored
     with a warning, exactly as the DALL-E back-end does.

Both modes share one set of weights: the img2img pipeline is built with
`from_pipe`, which rebinds the same transformer, VAE and text encoders into the
other class rather than loading a second copy.
"""

from __future__ import annotations

import time
from pathlib import Path

from PIL import Image as PILImage

from . import config
from .models import StructuredRadiologyPrompt
from .sd3_prompt_adapter import build_negative_prompt, split_prompt_for_sd3

VALID_MODES = ("txt2img", "img2img")

# mode → loaded diffusers pipeline. Populated by load_sd3_pipeline().
_PIPELINE_CACHE: dict[str, object] = {}


# ─── Dimension helpers ───────────────────────────────────────────────────────

def compute_sd3_dimensions(width: int, height: int) -> tuple[int, int]:
    """
    Pick the SD3 generation size closest to the source image's aspect ratio.

    SD3 takes explicit height/width rather than a ratio string, and both sides
    must be multiples of 16. config.SD3_SUPPORTED_DIMENSIONS holds ~1MP buckets
    satisfying that; this returns the (width, height) of the nearest one.
    """
    from .image_generator import compute_best_aspect_ratio

    ratios = {
        key: w / h for key, (w, h) in config.SD3_SUPPORTED_DIMENSIONS.items()
    }
    best_key = compute_best_aspect_ratio(width, height, ratios=ratios)
    return config.SD3_SUPPORTED_DIMENSIONS[best_key]


# ─── Mode resolution and init images ─────────────────────────────────────────

def _resolve_mode(mode: str | None, image_paths: list[Path] | None) -> str:
    """
    Determine the effective generation mode.

    img2img needs a reference image. Many Indiana records have no usable
    projection file, and the orchestrator passes image_paths=None for those, so
    img2img degrades to txt2img for that record rather than raising — a single
    missing PNG must never abort a long batch. The caller is expected to record
    the returned value, not the requested one, in metadata.
    """
    effective = (mode or config.SD3_MODE).lower()

    if effective not in VALID_MODES:
        raise ValueError(
            f"Unknown SD3 mode: {effective}. Use 'txt2img' or 'img2img'."
        )

    if effective == "img2img" and not image_paths:
        return "txt2img"

    return effective


def _load_init_image(path: Path, size: tuple[int, int]):
    """
    Load a reference X-ray as an SD3-ready init image.

    Converts to RGB (X-rays are 8-bit greyscale but the VAE needs 3 channels)
    and resizes to the target bucket. Source images are ~2000x2500, far above
    SD3's ~1MP operating point; passing them through unresized risks OOM and
    degrades output.
    """
    img = PILImage.open(path).convert("RGB")
    return img.resize(size, PILImage.LANCZOS)


# ─── Pipeline loading ────────────────────────────────────────────────────────

def _resolve_dtype(torch):
    """Map config.SD3_DTYPE to a torch dtype, warning on unsupported bf16."""
    requested = config.SD3_DTYPE.lower()

    if requested in ("bfloat16", "bf16"):
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            if major < 8:
                print(
                    "  ⚠ bfloat16 requested but this GPU (compute capability "
                    f"{major}.x) lacks bf16 tensor cores — falling back to "
                    "float16 to avoid black output."
                )
                return torch.float16
        return torch.bfloat16

    return torch.float16


def _select_device(torch) -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_sd3_pipeline(mode: str | None = None):
    """
    Load and cache the SD3 pipeline for a mode.

    Idempotent and safe to call per image; the orchestrator calls it once before
    the record loop so the load cost is not charged to the first report.

    The img2img pipeline is derived from the txt2img one with `from_pipe`, so
    both modes share a single set of weights in VRAM and a single download.
    """
    effective = (mode or config.SD3_MODE).lower()
    if effective not in VALID_MODES:
        raise ValueError(f"Unknown SD3 mode: {effective}. Use 'txt2img' or 'img2img'.")

    if effective in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[effective]

    import torch
    from diffusers import StableDiffusion3Pipeline

    # txt2img is always loaded first; img2img is rebound from it below.
    if "txt2img" not in _PIPELINE_CACHE:
        dtype = _resolve_dtype(torch)
        device = _select_device(torch)

        load_kwargs: dict = {
            "torch_dtype": dtype,
            "variant": "fp16",
        }
        if config.HUGGINGFACE_TOKEN:
            load_kwargs["token"] = config.HUGGINGFACE_TOKEN
        if config.SD3_DROP_T5:
            print(
                "  ⚠ SD3_DROP_T5 is set — the T5 encoder is being skipped. This "
                "frees ~10GB of VRAM but leaves only the 77-token CLIP encoders, "
                "so long clinical prompts will be truncated."
            )
            load_kwargs["text_encoder_3"] = None
            load_kwargs["tokenizer_3"] = None

        print(
            f"🧠 Loading SD3 ({config.SD3_MODEL_ID}, dtype={dtype}, device={device}) ..."
        )
        started = time.time()

        try:
            pipe = StableDiffusion3Pipeline.from_pretrained(
                config.SD3_MODEL_ID, **load_kwargs
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {config.SD3_MODEL_ID}. This model is a gated "
                "HuggingFace repo — accept the license at "
                "huggingface.co/stabilityai/stable-diffusion-3-medium and set "
                f"HUGGINGFACE_TOKEN. Original error: {e}"
            ) from e

        # CPU offload manages device placement itself; calling .to(device) as
        # well would defeat it and reintroduce the full VRAM requirement.
        if config.SD3_ENABLE_CPU_OFFLOAD and device == "cuda":
            pipe.enable_model_cpu_offload()
            print("   → model CPU offload enabled")
        else:
            pipe = pipe.to(device)

        if config.SD3_ENABLE_VAE_SLICING:
            # StableDiffusion3Pipeline doesn't expose enable_vae_slicing() as a
            # pipeline-level convenience method on every diffusers release —
            # go straight to the VAE itself, which supports it universally.
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            elif hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
                pipe.vae.enable_slicing()
            else:
                print("   ⚠ VAE slicing not available on this pipeline/diffusers version — skipping")

        print(f"   → SD3 ready in {time.time() - started:.1f}s")
        _PIPELINE_CACHE["txt2img"] = pipe

    if effective == "img2img" and "img2img" not in _PIPELINE_CACHE:
        from diffusers import StableDiffusion3Img2ImgPipeline

        # from_pipe reuses the loaded components — no second download, no
        # second copy of the weights in VRAM.
        _PIPELINE_CACHE["img2img"] = StableDiffusion3Img2ImgPipeline.from_pipe(
            _PIPELINE_CACHE["txt2img"]
        )
        print("   → img2img pipeline bound (weights shared with txt2img)")

    return _PIPELINE_CACHE[effective]


# ─── Generation ──────────────────────────────────────────────────────────────

def generate_image_sd3(
    prompt: str,
    uid: int,
    image_paths: list[Path] | None = None,
    view_suffix: str | None = None,
    source_dimensions: tuple[int, int] | None = None,
    max_retries: int = 3,
    structured_prompt: StructuredRadiologyPrompt | None = None,
    mode: str | None = None,
) -> Path:
    """
    Generate an image with Stable Diffusion 3.

    Parameters
    ----------
    prompt            : the formatted text prompt
    uid               : report UID, used for the output filename
    image_paths       : reference image paths. Used as the init image in
                        img2img mode; ignored with a warning in txt2img.
    view_suffix       : appended to the filename for multi-view reports
    source_dimensions : (width, height) of the original image; selects the
                        closest SD3 dimension bucket when provided
    structured_prompt : the StructuredRadiologyPrompt the text came from, used
                        to build a cleaner CLIP prompt when available
    mode              : "txt2img" or "img2img" (overrides config)

    Returns
    -------
    Path to the saved PNG file.
    """
    import torch

    effective_mode = _resolve_mode(mode, image_paths)

    if effective_mode == "txt2img" and image_paths:
        print(
            f"  ⚠ SD3 txt2img mode ignores the {len(image_paths)} reference "
            "image(s); set SD3_MODE=img2img to condition on them."
        )
    elif (mode or config.SD3_MODE).lower() == "img2img" and effective_mode == "txt2img":
        print(
            f"  ⚠ uid={uid}: img2img requested but no reference image available "
            "— falling back to txt2img for this record."
        )

    # ── Resolve generation geometry ──────────────────────────────────────
    if source_dimensions:
        width, height = compute_sd3_dimensions(*source_dimensions)
        print(
            f"  📐 SD3 size matched: {width}×{height} "
            f"(source: {source_dimensions[0]}×{source_dimensions[1]})"
        )
    else:
        width, height = config.SD3_WIDTH, config.SD3_HEIGHT

    # ── Build the two text channels ──────────────────────────────────────
    clip_prompt, t5_prompt = split_prompt_for_sd3(prompt, structured_prompt)
    negative = build_negative_prompt(config.SD3_NEGATIVE_PROMPT)

    pipe = load_sd3_pipeline(effective_mode)

    filename = f"{uid}_{view_suffix}.png" if view_suffix else f"{uid}.png"
    output_path = config.IMAGES_DIR / filename

    # Seeding per-uid keeps runs reproducible while keeping every record
    # distinct, and makes a txt2img/img2img pair directly comparable.
    if config.SD3_SEED is not None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(config.SD3_SEED + uid)
    else:
        generator = None

    common: dict = {
        "prompt": clip_prompt,
        "prompt_2": clip_prompt,
        "prompt_3": t5_prompt,
        "negative_prompt": negative,
        "negative_prompt_2": negative,
        "negative_prompt_3": negative,
        "num_inference_steps": config.SD3_STEPS,
        "guidance_scale": config.SD3_GUIDANCE_SCALE,
        "max_sequence_length": config.SD3_MAX_SEQUENCE_LENGTH,
        "generator": generator,
    }

    for attempt in range(1, max_retries + 1):
        try:
            if effective_mode == "txt2img":
                result = pipe(**common, height=height, width=width)
            else:
                # height/width are not accepted by the img2img pipeline —
                # geometry comes from the init image, which is resized to the
                # same bucket the txt2img path would have used.
                init_image = _load_init_image(image_paths[0], (width, height))
                result = pipe(
                    **common,
                    image=init_image,
                    strength=config.SD3_IMG2IMG_STRENGTH,
                )

            image = result.images[0]
            image.save(output_path)

            extra = (
                f", strength={config.SD3_IMG2IMG_STRENGTH}"
                if effective_mode == "img2img"
                else ""
            )
            print(
                f"  ✓ Image saved: {output_path.name} "
                f"(SD3 {effective_mode}, steps={config.SD3_STEPS}, "
                f"cfg={config.SD3_GUIDANCE_SCALE}{extra}, "
                f"output: {image.size[0]}×{image.size[1]})"
            )
            return output_path

        except Exception as e:
            # OOM is the one failure a retry can genuinely fix, once the
            # allocator has released the failed attempt's blocks.
            if _is_oom(e) and torch.cuda.is_available():
                torch.cuda.empty_cache()

            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  ⚠ SD3 attempt {attempt} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"SD3 generation failed for uid={uid} after "
                    f"{max_retries} attempts: {e}"
                ) from e

    return output_path  # unreachable but satisfies type checker


def _is_oom(exc: Exception) -> bool:
    """Detect CUDA out-of-memory without importing torch at module level."""
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()

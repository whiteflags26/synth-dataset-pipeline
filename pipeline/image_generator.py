"""
Image Generator — generates synthetic radiology images using DALL-E 3,
Google Vertex AI Gemini, or a local Stable Diffusion 3 model, driven by the
formatted text prompt.
Supports optional reference images as multimodal input for Gemini.
Supports source_dimensions to preserve the original image aspect ratio
natively (no post-processing resize) by passing the closest supported
aspect ratio to the generation API.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from io import BytesIO

from PIL import Image as PILImage

from . import config


# ─── Aspect Ratio Helpers ────────────────────────────────────────────────────

def compute_best_aspect_ratio(
    width: int,
    height: int,
    ratios: dict[str, float] | None = None,
) -> str:
    """
    Find the closest supported aspect ratio for the given pixel dimensions.

    Parameters
    ----------
    width   : source image width in pixels
    height  : source image height in pixels
    ratios  : mapping of ratio strings to decimal values.
              Defaults to config.GEMINI_SUPPORTED_ASPECT_RATIOS.

    Returns
    -------
    The ratio string with the smallest absolute difference (e.g. "3:4").
    """
    if ratios is None:
        ratios = config.GEMINI_SUPPORTED_ASPECT_RATIOS

    source_ratio = width / height
    best_key = min(ratios, key=lambda k: abs(ratios[k] - source_ratio))
    return best_key


def _best_dalle_size(width: int, height: int) -> str:
    """Pick the closest DALL-E 3 size string for the given dimensions."""
    source_ratio = width / height
    sizes = config.DALLE_SUPPORTED_SIZES
    return min(sizes, key=lambda k: abs(sizes[k] - source_ratio))


# ─── DALL-E 3 ────────────────────────────────────────────────────────────────

def generate_image_dalle(
    prompt: str,
    uid: int,
    view_suffix: str | None = None,
    source_dimensions: tuple[int, int] | None = None,
    max_retries: int = 3,
) -> Path:
    """
    Generate an image with OpenAI DALL-E 3.

    Parameters
    ----------
    source_dimensions : optional (width, height) of the original image.
                        When provided, the closest DALL-E size is selected
                        automatically instead of using config.DALLE_IMAGE_SIZE.

    Returns the path to the saved PNG file.
    """
    from openai import OpenAI

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    filename = f"{uid}_{view_suffix}.png" if view_suffix else f"{uid}.png"
    output_path = config.IMAGES_DIR / filename

    # Determine size — use source dimensions when available
    if source_dimensions:
        dalle_size = _best_dalle_size(*source_dimensions)
        print(
            f"  📐 DALL-E size matched: {dalle_size} "
            f"(source: {source_dimensions[0]}×{source_dimensions[1]})"
        )
    else:
        dalle_size = config.DALLE_IMAGE_SIZE

    for attempt in range(1, max_retries + 1):
        try:
            response = client.images.generate(
                model=config.DALLE_MODEL,
                prompt=prompt,
                size=dalle_size,
                quality=config.DALLE_IMAGE_QUALITY,
                response_format="b64_json",
                n=1,
            )
            image_data = base64.b64decode(response.data[0].b64_json)
            img = PILImage.open(BytesIO(image_data))
            img.save(output_path)
            return output_path

        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  ⚠ DALL-E attempt {attempt} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"DALL-E generation failed for uid={uid} after {max_retries} attempts: {e}"
                ) from e

    return output_path  # unreachable but satisfies type checker


# ─── Google Vertex AI Gemini Image Generation ────────────────────────────────

def generate_image_gemini(
    prompt: str,
    uid: int,
    image_paths: list[Path] | None = None,
    view_suffix: str | None = None,
    source_dimensions: tuple[int, int] | None = None,
    max_retries: int = 3,
) -> Path:
    """
    Generate an image with Google Gemini's native image generation via Vertex AI.

    Uses the google.genai Client with vertexai=True and the
    gemini-2.5-flash-image model with response_modalities=["TEXT", "IMAGE"].

    Parameters
    ----------
    prompt            : the formatted text prompt
    uid               : report UID, used for output filename
    image_paths       : optional list of reference image Paths to include as
                        multimodal input. When provided, images are prepended
                        to the content as inline Parts before the text prompt.
    source_dimensions : optional (width, height) of the original image.
                        When provided, the closest supported aspect ratio is
                        passed via ImageConfig so the API generates at the
                        correct geometry — no post-processing resize needed.
    max_retries       : number of retry attempts on failure

    Returns
    -------
    Path to the saved PNG file.
    """
    from google import genai
    from google.genai import types as genai_types

    if config.IS_KAGGLE:
        # Kaggle's kaggle_gcp.py causes a circular import with Vertex AI.
        # Use plain API key auth instead.
        client = genai.Client(api_key=config.GOOGLE_API_KEY)
    else:
        client = genai.Client(
            vertexai=True,
            project=config.GCP_PROJECT_ID,
            location=config.GCP_IMAGE_LOCATION,
        )

    model_id = config.GEMINI_IMAGE_MODEL
    filename = f"{uid}_{view_suffix}.png" if view_suffix else f"{uid}.png"
    output_path = config.IMAGES_DIR / filename

    # ── Compute aspect ratio from source dimensions ───────────────────────
    aspect_ratio: str | None = None
    if source_dimensions:
        aspect_ratio = compute_best_aspect_ratio(*source_dimensions)
        print(
            f"  📐 Aspect ratio matched: {aspect_ratio} "
            f"(source: {source_dimensions[0]}×{source_dimensions[1]}, "
            f"ratio: {source_dimensions[0]/source_dimensions[1]:.3f})"
        )

    # ── Build multimodal contents ─────────────────────────────────────────
    if image_paths:
        contents: list = []
        for img_path in image_paths:
            with open(img_path, "rb") as f:
                img_bytes = f.read()
            # Determine MIME type from extension
            suffix = img_path.suffix.lower()
            mime_map = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
            mime_type = mime_map.get(suffix, "image/png")
            contents.append(
                genai_types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
            )
        # Append text prompt as a final Part
        contents.append(genai_types.Part.from_text(text=prompt))
        ref_count = len(image_paths)
    else:
        # Text-only: pass as a plain string (same as before)
        contents = prompt
        ref_count = 0

    # ── Build generation config ───────────────────────────────────────────
    gen_config_kwargs: dict = {
        "response_modalities": ["TEXT", "IMAGE"],
        "candidate_count": 1,
    }
    if aspect_ratio:
        gen_config_kwargs["image_config"] = genai_types.ImageConfig(
            aspect_ratio=aspect_ratio,
        )

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=genai_types.GenerateContentConfig(**gen_config_kwargs),
            )

            # Extract image data from response parts
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    img = PILImage.open(BytesIO(part.inline_data.data))
                    img.save(output_path)
                    ref_info = f", {ref_count} reference image(s)" if ref_count else ""
                    ar_info = f", aspect_ratio={aspect_ratio}" if aspect_ratio else ""
                    print(
                        f"  ✓ Image saved: {output_path.name} "
                        f"(model: {model_id}{ref_info}{ar_info}, "
                        f"output: {img.size[0]}×{img.size[1]})"
                    )
                    return output_path

            raise RuntimeError("No image data found in response")

        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  ⚠ Gemini attempt {attempt} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Gemini image generation failed for uid={uid} after {max_retries} attempts: {e}"
                ) from e

    return output_path  # unreachable


# ─── Dispatcher ──────────────────────────────────────────────────────────────

def generate_image(
    prompt: str,
    uid: int,
    generator: str | None = None,
    image_paths: list[Path] | None = None,
    view_suffix: str | None = None,
    source_dimensions: tuple[int, int] | None = None,
    structured_prompt=None,
    sd3_mode: str | None = None,
) -> Path:
    """
    Generate an image using the configured (or specified) backend.

    Parameters
    ----------
    prompt            : the formatted image generation prompt
    uid               : report UID, used for filename
    generator         : "dalle", "gemini", or "sd3" (overrides config)
    image_paths       : optional reference image paths. Used by Gemini as
                        multimodal input and by SD3 in img2img mode; ignored
                        by DALL-E and by SD3 in txt2img mode.
    source_dimensions : optional (width, height) of the original image.
                        Passed to the backend to generate at the closest
                        native aspect ratio — no post-processing resize.
    structured_prompt : optional StructuredRadiologyPrompt the text came from.
                        SD3 only, where it produces a cleaner CLIP prompt.
    sd3_mode          : "txt2img" or "img2img" (SD3 only, overrides config)

    Returns
    -------
    Path to the saved image file.
    """
    gen = (generator or config.IMAGE_GENERATOR).lower()

    if gen == "dalle":
        if image_paths:
            print(f"  ⚠ Reference images are not supported for DALL-E, ignoring {len(image_paths)} image(s).")
        return generate_image_dalle(
            prompt, uid, view_suffix=view_suffix,
            source_dimensions=source_dimensions,
        )
    elif gen == "gemini":
        return generate_image_gemini(
            prompt, uid, image_paths=image_paths,
            view_suffix=view_suffix,
            source_dimensions=source_dimensions,
        )
    elif gen == "sd3":
        from .sd3_generator import generate_image_sd3

        return generate_image_sd3(
            prompt, uid, image_paths=image_paths,
            view_suffix=view_suffix,
            source_dimensions=source_dimensions,
            structured_prompt=structured_prompt,
            mode=sd3_mode,
        )
    else:
        raise ValueError(
            f"Unknown image generator: {gen}. Use 'dalle', 'gemini', or 'sd3'."
        )

"""
Configuration module — loads environment variables and exposes pipeline settings.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ── Load .env ────────────────────────────────────────────────────────────────
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

# ── Environment detection ────────────────────────────────────────────────────
IS_KAGGLE: bool = "KAGGLE_KERNEL_RUN_TYPE" in os.environ

# ── API keys ─────────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN", "")

# ── Vertex AI settings ──────────────────────────────────────────────────────
GCP_PROJECT_ID: str = os.getenv("GCP_PROJECT_ID", "")
GCP_LOCATION: str = os.getenv("GCP_LOCATION", "us-central1")
GCP_IMAGE_LOCATION: str = os.getenv("GCP_IMAGE_LOCATION", "global")
VERTEX_ENDPOINT_ID: str = os.getenv("VERTEX_ENDPOINT_ID", "")

# ── Model settings ───────────────────────────────────────────────────────────
CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
IMAGE_GENERATOR: str = os.getenv("IMAGE_GENERATOR", "sd3")  # "dalle", "gemini", or "sd3"
GEMINI_IMAGE_MODEL: str = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
DALLE_MODEL: str = os.getenv("DALLE_MODEL", "dall-e-3")
DALLE_IMAGE_SIZE: str = os.getenv("DALLE_IMAGE_SIZE", "1024x1024")
DALLE_IMAGE_QUALITY: str = os.getenv("DALLE_IMAGE_QUALITY", "standard")

# ── Stable Diffusion 3 settings ──────────────────────────────────────────────
# Defaults follow Stability AI's recommended settings for SD3 Medium
# (28 steps, CFG 7.0, 1024x1024). The model is a gated HuggingFace repo:
# accept the license at huggingface.co/stabilityai/stable-diffusion-3-medium
# and set HUGGINGFACE_TOKEN before the first run.
SD3_MODEL_ID: str = os.getenv(
    "SD3_MODEL_ID", "stabilityai/stable-diffusion-3-medium-diffusers"
)
SD3_MODE: str = os.getenv("SD3_MODE", "txt2img")  # "txt2img" or "img2img"
SD3_STEPS: int = int(os.getenv("SD3_STEPS", "28"))
SD3_GUIDANCE_SCALE: float = float(os.getenv("SD3_GUIDANCE_SCALE", "7.0"))
SD3_HEIGHT: int = int(os.getenv("SD3_HEIGHT", "1024"))
SD3_WIDTH: int = int(os.getenv("SD3_WIDTH", "1024"))
SD3_DTYPE: str = os.getenv("SD3_DTYPE", "float16")  # "float16" or "bfloat16"
SD3_NEGATIVE_PROMPT: str = os.getenv("SD3_NEGATIVE_PROMPT", "")
SD3_MAX_SEQUENCE_LENGTH: int = int(os.getenv("SD3_MAX_SEQUENCE_LENGTH", "512"))
SD3_IMG2IMG_STRENGTH: float = float(os.getenv("SD3_IMG2IMG_STRENGTH", "0.75"))

_sd3_seed_raw: str = os.getenv("SD3_SEED", "").strip()
SD3_SEED: Optional[int] = int(_sd3_seed_raw) if _sd3_seed_raw else None

# Memory management. CPU offload keeps SD3 Medium inside a 16GB card at the
# cost of speed; dropping T5 frees ~10GB but leaves only the 77-token CLIP
# encoders, discarding the long-prompt path sd3_prompt_adapter is built for.
SD3_ENABLE_CPU_OFFLOAD: bool = os.getenv("SD3_ENABLE_CPU_OFFLOAD", "true").lower() == "true"
SD3_ENABLE_VAE_SLICING: bool = os.getenv("SD3_ENABLE_VAE_SLICING", "true").lower() == "true"
SD3_DROP_T5: bool = os.getenv("SD3_DROP_T5", "false").lower() == "true"

# ── Aspect ratio mappings ────────────────────────────────────────────────
# Gemini: ratio string → decimal value (width / height) for closest-match
GEMINI_SUPPORTED_ASPECT_RATIOS: dict[str, float] = {
    "1:1":  1.0,
    "4:3":  4 / 3,
    "3:4":  3 / 4,
    "16:9": 16 / 9,
    "9:16": 9 / 16,
    "3:2":  3 / 2,
    "2:3":  2 / 3,
    "5:4":  5 / 4,
    "4:5":  4 / 5,
    "21:9": 21 / 9,
}

# SD3: ratio string → (width, height). Each bucket is ~1 megapixel with both
# sides a multiple of 16, as required by the VAE (factor 8) and patch size (2).
SD3_SUPPORTED_DIMENSIONS: dict[str, tuple[int, int]] = {
    "1:1":  (1024, 1024),
    "4:3":  (1152,  896),
    "3:4":  ( 896, 1152),
    "3:2":  (1216,  832),
    "2:3":  ( 832, 1216),
    "16:9": (1344,  768),
    "9:16": ( 768, 1344),
    "5:4":  (1088,  896),
    "4:5":  ( 896, 1088),
}

# DALL-E 3: aspect ratio decimal → closest supported size string
DALLE_SUPPORTED_SIZES: dict[str, float] = {
    "1024x1024": 1.0,      # square
    "1792x1024": 1792 / 1024,  # landscape (~1.75)
    "1024x1792": 1024 / 1792,  # portrait  (~0.571)
}

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
IMAGES_DIR = OUTPUT_DIR / "images"
METADATA_FILE = OUTPUT_DIR / "metadata.json"

# Ensure output dirs exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

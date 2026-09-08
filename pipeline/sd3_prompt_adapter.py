"""
SD3 Prompt Adapter — reshapes the long, multi-section image prompt produced by
image_prompt_formatter into the three-channel form Stable Diffusion 3 expects.

SD3 encodes text with three encoders: CLIP-L and CLIP-G (both hard-truncate at
77 tokens) and T5-XXL (up to 512). The formatted radiology prompt is ~400-500
tokens, so feeding it directly means the CLIP encoders see only the leading
boilerplate and never reach the clinical findings.

StableDiffusion3Pipeline accepts `prompt` / `prompt_2` / `prompt_3` as separate
strings routed to CLIP-L / CLIP-G / T5. This module builds a compact
visual-essence prompt for the CLIP channels and passes the full structured text
to T5, and lifts the negative constraints out of the positive prompt into a
proper `negative_prompt` (stating prohibitions positively inside a diffusion
prompt reliably summons what they forbid).
"""

from __future__ import annotations

import re

from .models import StructuredRadiologyPrompt

# Conservative stand-in for CLIP's 77-token limit. A word is usually >= 1 token,
# and the tokenizer also spends tokens on start/end markers and punctuation.
CLIP_WORD_BUDGET = 60

# Sections removed from the T5 prompt: reference-image instructions are
# meaningless to SD3 (the image is passed as a latent, not described), and the
# negative constraints move to the negative_prompt channel.
_T5_DROPPED_SECTIONS = ("REFERENCE IMAGES", "NEGATIVE CONSTRAINTS — STRICT")

# Diffusion negatives want comma-separated noun phrases, not prose sentences.
_BASE_NEGATIVE = (
    "text, letters, words, numbers, labels, captions, annotations, watermark, "
    "signature, overlay, header bar, report panel, UI elements, scrollbar, "
    "border, frame, color, colour, photograph, illustration, cartoon, drawing, "
    "blurry, low quality, deformed anatomy, duplicated organs, extra limbs"
)

_SECTION_RE_TMPL = r"\[{name}\]\n(.*?)(?=\n\n\[|\Z)"


# ─── Private helpers ─────────────────────────────────────────────────────────

def _extract_section(text: str, name: str) -> str | None:
    """Return the body of a [SECTION NAME] block, or None when absent."""
    pattern = _SECTION_RE_TMPL.format(name=re.escape(name))
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else None


def _strip_section(text: str, name: str) -> str:
    """Remove a whole [SECTION NAME] block, header and body, from the prompt."""
    pattern = r"\[" + re.escape(name) + r"\]\n.*?(?=\n\n\[|\Z)"
    stripped = re.sub(pattern, "", text, flags=re.DOTALL)
    # Collapse the blank-line runs left behind by the removal.
    return re.sub(r"\n{3,}", "\n\n", stripped).strip()


def _truncate_to_words(text: str, max_words: int) -> str:
    """Cut text to at most max_words whitespace-separated words."""
    words = text.split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _first_clause(text: str, max_words: int = 24) -> str:
    """
    Take the leading clinical statement from a findings paragraph.

    Findings are written as sentences; the first one or two carry the salient
    visual content, and the rest are usually negations ("no pneumothorax")
    that belong in the T5 channel rather than the CLIP summary.
    """
    cleaned = " ".join(text.split())
    sentences = re.split(r"(?<=[.;])\s+", cleaned)
    out: list[str] = []
    for sentence in sentences:
        candidate = out + [sentence]
        if len(" ".join(candidate).split()) > max_words and out:
            break
        out = candidate
    joined = " ".join(out).strip().rstrip(".;,")
    return _truncate_to_words(joined, max_words)


def _clip_from_structured(sp: StructuredRadiologyPrompt) -> str:
    """Assemble the CLIP prompt from the structured fields."""
    parts: list[str] = []

    parts.append(sp.modality or "Chest X-ray radiograph")

    view = sp.view or sp.plane_view
    if view:
        parts.append(f"{view} view")

    if sp.anatomical_region:
        parts.append(sp.anatomical_region)

    if sp.findings:
        parts.append(_first_clause(sp.findings))

    parts.append("grayscale diagnostic radiograph")

    return _truncate_to_words(", ".join(p for p in parts if p), CLIP_WORD_BUDGET)


def _clip_from_text(prompt_text: str) -> str:
    """
    Fallback CLIP prompt built by mining the formatted text.

    Used when no StructuredRadiologyPrompt is available (for example when
    prompts are replayed from an existing metadata.json).
    """
    parts: list[str] = []

    modality = _extract_section(prompt_text, "MODALITY CONDITIONING")
    if modality:
        # Drop the "Radiology-grade " lead-in and the trailing protocol boilerplate.
        lead = modality.split(", clinical")[0]
        parts.append(re.sub(r"^Radiology-grade\s+", "", lead).strip())

    anatomical = _extract_section(prompt_text, "ANATOMICAL PROMPTING")
    if anatomical:
        match = re.match(r"Focused on ([^,]+)", anatomical)
        if match:
            parts.append(match.group(1).strip())

    clinical = _extract_section(prompt_text, "TEXTUAL PROMPTING — CLINICAL DESCRIPTION")
    if clinical:
        body = re.sub(r"^Radiological findings include:\s*", "", clinical)
        parts.append(_first_clause(body))

    if not parts:
        parts.append("Chest X-ray radiograph")

    parts.append("grayscale diagnostic radiograph")

    return _truncate_to_words(", ".join(p for p in parts if p), CLIP_WORD_BUDGET)


# ─── Public entry points ─────────────────────────────────────────────────────

def split_prompt_for_sd3(
    prompt_text: str,
    sp: StructuredRadiologyPrompt | None = None,
) -> tuple[str, str]:
    """
    Split a formatted image prompt into SD3's two text channels.

    Parameters
    ----------
    prompt_text : the full output of image_prompt_formatter.format_image_prompt
    sp          : the structured prompt it came from, when available. Produces a
                  cleaner CLIP prompt; falls back to mining prompt_text when None.

    Returns
    -------
    (clip_prompt, t5_prompt)
        clip_prompt : compact visual essence, <= CLIP_WORD_BUDGET words, for the
                      `prompt` and `prompt_2` arguments (CLIP-L / CLIP-G).
        t5_prompt   : the full structured text minus reference-image and negative
                      sections, for the `prompt_3` argument (T5-XXL).
    """
    t5_prompt = prompt_text
    for section in _T5_DROPPED_SECTIONS:
        t5_prompt = _strip_section(t5_prompt, section)

    if sp is not None:
        clip_prompt = _clip_from_structured(sp)
    else:
        clip_prompt = _clip_from_text(prompt_text)

    return clip_prompt, t5_prompt


def build_negative_prompt(extra: str | None = None) -> str:
    """
    Build the SD3 negative prompt.

    The [NEGATIVE CONSTRAINTS — STRICT] section of the formatted prompt is prose
    aimed at instruction-following models. SD3 has a real negative channel, so
    the prohibitions are restated here as comma-separated noun phrases and
    removed from the positive prompt by split_prompt_for_sd3.

    Parameters
    ----------
    extra : additional comma-separated negatives appended to the defaults,
            typically config.SD3_NEGATIVE_PROMPT.
    """
    if extra and extra.strip():
        return f"{_BASE_NEGATIVE}, {extra.strip()}"
    return _BASE_NEGATIVE

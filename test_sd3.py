"""
Smoke checks for the SD3 back-end.

Runs without a GPU, without torch or diffusers installed, and without
downloading a model — check 3 depends on that being true. Plain script in the
style of test_pipeline.py / test_aspect_ratio.py:

    python test_sd3.py
"""

import sys
import tempfile
from pathlib import Path

from PIL import Image as PILImage

from pipeline import config
from pipeline.image_generator import generate_image
from pipeline.image_prompt_formatter import format_image_prompt
from pipeline.models import ImageProjection, StructuredRadiologyPrompt
from pipeline.sd3_generator import (
    _load_init_image,
    _resolve_mode,
    compute_sd3_dimensions,
)
from pipeline.sd3_prompt_adapter import (
    CLIP_WORD_BUDGET,
    build_negative_prompt,
    split_prompt_for_sd3,
)

failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  ✓ {label}")
    else:
        print(f"  ✗ {label}{f' — {detail}' if detail else ''}")
        failures.append(label)


def sample_prompt() -> StructuredRadiologyPrompt:
    sp = StructuredRadiologyPrompt(
        modality="Chest X-ray",
        anatomical_region="Chest, lungs and mediastinum",
        plane_view="PA and lateral",
        patient_demographics="55-year-old male",
        findings=(
            "The cardiomediastinal silhouette is within normal limits. "
            "Mild bibasilar atelectasis is present. There is no pneumothorax "
            "or pleural effusion. The osseous structures are intact."
        ),
        imaging_characteristics="Normal tissue density and contrast",
        view="PA",
        reference_images=[
            ImageProjection(filename="1_IM-0001-3001.png", projection="Frontal")
        ],
    )
    sp.source_dimensions = (2048, 2496)
    sp.matched_aspect_ratio = "4:5"
    return sp


# ─── 1. Dimension buckets ────────────────────────────────────────────────────

print("\n[1] compute_sd3_dimensions")

# (source_w, source_h) -> expected (w, h). Portrait shapes are the ones that
# actually occur in the Indiana chest X-ray set.
DIMENSION_CASES = [
    ((2048, 2496), (896, 1088)),    # 0.820 -> 4:5
    ((1024, 1024), (1024, 1024)),   # square
    ((2500, 2048), (1088, 896)),    # 1.221 -> 5:4 (1.214), nearer than 4:3 (1.333)
    ((2400, 1800), (1152, 896)),    # 1.333 -> 4:3 exactly
    ((1600, 2400), (832, 1216)),    # 0.667 -> 2:3
    ((1920, 1080), (1344, 768)),    # 1.778 -> 16:9
    ((2021, 2418), (896, 1088)),    # a real Indiana frontal, 0.836 -> 4:5
]

for (src, expected) in DIMENSION_CASES:
    got = compute_sd3_dimensions(*src)
    check(
        f"{src[0]}×{src[1]} → {expected[0]}×{expected[1]}",
        got == expected,
        f"got {got[0]}×{got[1]}",
    )

# Every bucket must be VAE-compatible or diffusers will error at runtime.
for key, (w, h) in config.SD3_SUPPORTED_DIMENSIONS.items():
    check(f"bucket {key} sides divisible by 16", w % 16 == 0 and h % 16 == 0)


# ─── 2. Prompt splitting ─────────────────────────────────────────────────────

print("\n[2] split_prompt_for_sd3")

sp = sample_prompt()
full_prompt = format_image_prompt(sp)
clip_prompt, t5_prompt = split_prompt_for_sd3(full_prompt, sp)

check(
    f"clip_prompt within {CLIP_WORD_BUDGET}-word budget",
    len(clip_prompt.split()) <= CLIP_WORD_BUDGET,
    f"got {len(clip_prompt.split())} words",
)
check(
    "clip_prompt carries clinical findings, not just boilerplate",
    "cardiomediastinal" in clip_prompt.lower(),
    f"got: {clip_prompt!r}",
)
check("clip_prompt names the modality", "chest x-ray" in clip_prompt.lower())
check("clip_prompt names the view", "pa view" in clip_prompt.lower())

check(
    "t5_prompt drops [NEGATIVE CONSTRAINTS",
    "[NEGATIVE CONSTRAINTS" not in t5_prompt,
)
check("t5_prompt drops [REFERENCE IMAGES]", "[REFERENCE IMAGES]" not in t5_prompt)
check("t5_prompt keeps [TEXTUAL PROMPTING", "[TEXTUAL PROMPTING" in t5_prompt)
check("t5_prompt keeps [ANATOMICAL PROMPTING]", "[ANATOMICAL PROMPTING]" in t5_prompt)
check("t5_prompt keeps [MODALITY CONDITIONING]", "[MODALITY CONDITIONING]" in t5_prompt)
check("t5_prompt is shorter than the original", len(t5_prompt) < len(full_prompt))
check("t5_prompt has no blank-line runs", "\n\n\n" not in t5_prompt)

# The fallback path: no structured prompt available (replayed from metadata).
clip_fb, t5_fb = split_prompt_for_sd3(full_prompt, None)
check(
    f"fallback clip_prompt within budget",
    len(clip_fb.split()) <= CLIP_WORD_BUDGET,
    f"got {len(clip_fb.split())} words",
)
check(
    "fallback clip_prompt carries findings",
    "cardiomediastinal" in clip_fb.lower(),
    f"got: {clip_fb!r}",
)
check("fallback t5_prompt matches structured path", t5_fb == t5_prompt)

# A minimal prompt must not crash the adapter.
bare = StructuredRadiologyPrompt(findings="Bilateral pleural effusions.")
bare_clip, bare_t5 = split_prompt_for_sd3(format_image_prompt(bare), bare)
check("bare prompt yields a non-empty clip_prompt", bool(bare_clip.strip()))
check("bare prompt yields a non-empty t5_prompt", bool(bare_t5.strip()))


# ─── 3. Negative prompt ──────────────────────────────────────────────────────

print("\n[3] build_negative_prompt")

negative = build_negative_prompt(None)
check("negative names text", "text" in negative)
check("negative is comma-separated, not prose", "Do NOT" not in negative)
check("negative appends extras", "custom term" in build_negative_prompt("custom term"))
check("negative ignores blank extras", build_negative_prompt("   ") == negative)


# ─── 4. Deferred imports (no torch installed) ────────────────────────────────

print("\n[4] deferred SDK imports")

check(
    "pipeline.sd3_generator imports without torch",
    "torch" not in sys.modules,
    "torch was imported at module load — move it inside the function",
)
check(
    "pipeline.sd3_generator imports without diffusers",
    "diffusers" not in sys.modules,
    "diffusers was imported at module load",
)


# ─── 5. Mode resolution ──────────────────────────────────────────────────────

print("\n[5] _resolve_mode")

ref = [Path("some_xray.png")]

check("img2img without a reference downgrades", _resolve_mode("img2img", None) == "txt2img")
check("img2img with an empty list downgrades", _resolve_mode("img2img", []) == "txt2img")
check("img2img with a reference stays", _resolve_mode("img2img", ref) == "img2img")
check("txt2img with a reference stays", _resolve_mode("txt2img", ref) == "txt2img")
check("txt2img without a reference stays", _resolve_mode("txt2img", None) == "txt2img")
check("case is normalised", _resolve_mode("IMG2IMG", ref) == "img2img")

_saved_mode = config.SD3_MODE
config.SD3_MODE = "img2img"
check("None falls back to config", _resolve_mode(None, ref) == "img2img")
config.SD3_MODE = _saved_mode

try:
    _resolve_mode("inpaint", ref)
    check("unknown mode raises ValueError", False, "no exception raised")
except ValueError:
    check("unknown mode raises ValueError", True)


# ─── 6. Init image loading ───────────────────────────────────────────────────

print("\n[6] _load_init_image")

with tempfile.TemporaryDirectory() as tmp:
    # X-rays on disk are 8-bit greyscale at roughly this size.
    grey_path = Path(tmp) / "xray.png"
    PILImage.new("L", (2048, 2496), color=128).save(grey_path)

    init = _load_init_image(grey_path, (896, 1088))
    check("greyscale converted to RGB", init.mode == "RGB", f"got {init.mode}")
    check("resized to the bucket", init.size == (896, 1088), f"got {init.size}")


# ─── 7. Dispatcher routing ───────────────────────────────────────────────────

print("\n[7] generate_image dispatch")

import pipeline.sd3_generator as sd3_mod

recorded: dict = {}
_real = sd3_mod.generate_image_sd3


def _fake(prompt, uid, **kwargs):
    recorded.update({"prompt": prompt, "uid": uid, **kwargs})
    return Path("/tmp/fake.png")


sd3_mod.generate_image_sd3 = _fake
try:
    generate_image(
        "a prompt",
        42,
        generator="sd3",
        image_paths=ref,
        view_suffix="PA",
        source_dimensions=(2048, 2496),
        structured_prompt=sp,
        sd3_mode="img2img",
    )
    check("generator='sd3' reaches the SD3 back-end", recorded.get("uid") == 42)
    check("sd3_mode is forwarded as mode", recorded.get("mode") == "img2img")
    check("structured_prompt is forwarded", recorded.get("structured_prompt") is sp)
    check("image_paths are forwarded", recorded.get("image_paths") == ref)
    check("source_dimensions are forwarded", recorded.get("source_dimensions") == (2048, 2496))
finally:
    sd3_mod.generate_image_sd3 = _real

try:
    generate_image("a prompt", 1, generator="midjourney")
    check("unknown generator raises ValueError", False, "no exception raised")
except ValueError as e:
    check("unknown generator raises ValueError", "sd3" in str(e))


# ─── 8. run_from_prompts (two-session workflow) ──────────────────────────────

print("\n[8] run_from_prompts")

import json

from pipeline import pipeline as pl

_fixture = [
    {
        "uid": 1,
        "reference_images": [{"filename": "ref.png", "projection": "Frontal"}],
        "source_dimensions": {"width": 2048, "height": 2496},
        "views": [
            {"view": "PA", "image_prompt": "[MODALITY CONDITIONING]\nChest X-ray, PA."},
            {"view": "Lateral", "image_prompt": "[MODALITY CONDITIONING]\nChest X-ray, lateral."},
        ],
    },
    {"uid": 2, "views": [{"view": None, "image_prompt": "[MODALITY CONDITIONING]\nChest X-ray."}]},
    {"uid": 3, "error_prompt": "LLM failed"},  # no views — must be skipped
]

_saved = (config.OUTPUT_DIR, config.IMAGES_DIR, config.METADATA_FILE, pl.generate_image)

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    (tmp_path / "meta.json").write_text(json.dumps(_fixture))
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    PILImage.new("L", (2048, 2496), color=128).save(img_dir / "ref.png")

    config.OUTPUT_DIR = tmp_path / "out"
    config.IMAGES_DIR = config.OUTPUT_DIR / "images"
    config.METADATA_FILE = config.OUTPUT_DIR / "metadata.json"

    calls: list[dict] = []

    def _fake_generate(prompt, uid, **kwargs):
        calls.append(
            {
                "uid": uid,
                "view": kwargs.get("view_suffix"),
                "refs": kwargs.get("image_paths"),
                "dims": kwargs.get("source_dimensions"),
            }
        )
        suffix = kwargs.get("view_suffix")
        out = config.IMAGES_DIR / (f"{uid}_{suffix}.png" if suffix else f"{uid}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        PILImage.new("L", (8, 8)).save(out)
        return out

    pl.generate_image = _fake_generate
    try:
        results = pl.run_from_prompts(
            tmp_path / "meta.json",
            generator="dalle",
            images_dir=img_dir,
            output_subdir="replay",
        )
    finally:
        pl.generate_image = _saved[3]

    check("generates one image per view", len(calls) == 3, f"got {len(calls)}")
    check("skips entries with no views", all(c["uid"] != 3 for c in calls))
    check(
        "preserves view suffixes",
        [c["view"] for c in calls] == ["PA", "Lateral", None],
        f"got {[c['view'] for c in calls]}",
    )
    check(
        "resolves reference images from metadata",
        bool(calls[0]["refs"]) and calls[0]["refs"][0].name == "ref.png",
    )
    check("passes no refs when the record has none", calls[2]["refs"] is None)
    check("replays source_dimensions", calls[0]["dims"] == (2048, 2496))
    check("returns one entry per report", len(results) == 2)
    check("honours output_subdir", config.IMAGES_DIR.name == "replay")
    check("writes metadata", config.METADATA_FILE.exists())

    written = json.loads(config.METADATA_FILE.read_text())
    check("metadata records the generator", written[0]["generator"] == "dalle")
    check("metadata records image_path", "image_path" in written[0]["views"][0])

config.OUTPUT_DIR, config.IMAGES_DIR, config.METADATA_FILE, pl.generate_image = _saved


# ─── 9. _load_completed_uids ─────────────────────────────────────────────────

print("\n[9] _load_completed_uids")

from pipeline.pipeline import _load_completed_uids

with tempfile.TemporaryDirectory() as tmp:
    missing_path = Path(tmp) / "does_not_exist.json"
    by_uid, completed = _load_completed_uids(missing_path)
    check("missing file yields no completed uids", completed == set())
    check("missing file yields an empty by_uid map", by_uid == {})

    meta_path = Path(tmp) / "meta.json"
    meta_path.write_text(
        json.dumps(
            [
                # Fully successful, single view — done.
                {"uid": 1, "views": [{"view": None, "image_path": "1.png"}]},
                # Fully successful, multi-view — done.
                {
                    "uid": 2,
                    "views": [
                        {"view": "PA", "image_path": "2_PA.png"},
                        {"view": "Lateral", "image_path": "2_Lateral.png"},
                    ],
                },
                # Prompt extraction failed — retry.
                {"uid": 3, "error_prompt": "LLM timeout"},
                # One view succeeded, one failed — retry (no partial patching).
                {
                    "uid": 4,
                    "views": [
                        {"view": "PA", "image_path": "4_PA.png"},
                        {"view": "Lateral", "error_image": "OOM"},
                    ],
                },
                # No views at all (e.g. skip-images run) — retry.
                {"uid": 5, "views": []},
            ]
        )
    )

    by_uid, completed = _load_completed_uids(meta_path)
    check("finds every entry regardless of completion", set(by_uid) == {1, 2, 3, 4, 5})
    check("single-view success counts as complete", 1 in completed)
    check("multi-view success counts as complete", 2 in completed)
    check("error_prompt entry is not complete", 3 not in completed)
    check("partially failed multi-view entry is not complete", 4 not in completed)
    check("empty-views entry is not complete", 5 not in completed)
    check("exactly the two done uids are returned", completed == {1, 2})


# ─── 10. run_from_prompts resume behaviour ───────────────────────────────────

print("\n[10] run_from_prompts(resume=True)")

_saved10 = (config.OUTPUT_DIR, config.IMAGES_DIR, config.METADATA_FILE, pl.generate_image)

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    (tmp_path / "prompts.json").write_text(
        json.dumps(
            [
                {"uid": 1, "views": [{"view": None, "image_prompt": "prompt one"}]},
                {"uid": 2, "views": [{"view": None, "image_prompt": "prompt two"}]},
                {"uid": 3, "views": [{"view": None, "image_prompt": "prompt three"}]},
            ]
        )
    )

    config.OUTPUT_DIR = tmp_path / "out"
    config.IMAGES_DIR = config.OUTPUT_DIR / "images"
    config.METADATA_FILE = config.OUTPUT_DIR / "metadata.json"
    config.IMAGES_DIR.mkdir(parents=True)

    # Simulate a prior run that crashed after uid=1 succeeded and uid=2 failed.
    config.METADATA_FILE.write_text(
        json.dumps(
            [
                {"uid": 1, "views": [{"view": None, "image_path": "1.png"}]},
                {"uid": 2, "views": [{"view": None, "error_image": "OOM"}]},
            ]
        )
    )

    generate_calls: list[int] = []

    def _fake_generate(prompt, uid, **kwargs):
        generate_calls.append(uid)
        out = config.IMAGES_DIR / f"{uid}.png"
        PILImage.new("L", (8, 8)).save(out)
        return out

    _saved_fn = pl.generate_image
    pl.generate_image = _fake_generate
    try:
        results = pl.run_from_prompts(
            tmp_path / "prompts.json", generator="dalle", resume=True
        )
    finally:
        pl.generate_image = _saved_fn

    check(
        "already-complete uid=1 is not regenerated",
        1 not in generate_calls,
        f"generate_calls={generate_calls}",
    )
    check(
        "previously-failed uid=2 is retried",
        2 in generate_calls,
        f"generate_calls={generate_calls}",
    )
    check("new uid=3 is generated", 3 in generate_calls)
    check("final results include all three uids", {r["uid"] for r in results} == {1, 2, 3})

    written = json.loads(config.METADATA_FILE.read_text())
    check(
        "uid=1's carried-over entry is unchanged",
        next(e for e in written if e["uid"] == 1)["views"][0]["image_path"] == "1.png",
    )
    check(
        "uid=2 now has an image_path instead of an error",
        "image_path" in next(e for e in written if e["uid"] == 2)["views"][0],
    )

config.OUTPUT_DIR, config.IMAGES_DIR, config.METADATA_FILE, pl.generate_image = _saved10


# ─── Summary ─────────────────────────────────────────────────────────────────

print()
if failures:
    print(f"❌ {len(failures)} check(s) failed:")
    for f in failures:
        print(f"   - {f}")
    sys.exit(1)

print("✅ All SD3 checks passed.")

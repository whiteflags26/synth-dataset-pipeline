# PLAN — SD3 Image Back-End + OpenAI Prompt Extraction + Kaggle GPU Runner

**Goal.** Add `sd3` as a third image-generation back-end alongside `dalle` and `gemini`, supporting
**both text-to-image and image-to-image** modes; switch structured prompt extraction to **OpenAI
`gpt-4.1-mini`**; ship a Kaggle notebook that clones
`https://github.com/whiteflags26/synth-dataset-pipeline.git` and generates images on a cloud GPU.

**Net effect:** the pipeline stops depending on Google Cloud entirely. OpenAI handles the text stage,
SD3 runs locally on the GPU. One API key, one HF token, no service account.

All decisions below are settled — this document is meant to be executed top to bottom.

---

## 1. Settled decisions

### 1.1 Prompt extraction model: `gpt-4.1-mini`

`CHAT_MODEL=gpt-4.1-mini`. **This requires no code changes** — `_is_vertex_model()` in
[prompt_builder.py:60](pipeline/prompt_builder.py#L60) only matches `gemini/medlm/med-palm/claude/google/`
prefixes, so any `gpt-*` name already falls through to the `ChatOpenAI` branch at
[prompt_builder.py:118](pipeline/prompt_builder.py#L118) with `with_structured_output(...)` wired up.

| Model | ~$/1M in | ~$/1M out | Est. cost, 1000 reports | Verdict |
|---|---|---|---|---|
| **`gpt-4.1-mini`** | $0.40 | $1.60 | **~$0.65** | **Use this.** Strong schema adherence, non-reasoning, supports `temperature` |
| `gpt-4.1-nano` | $0.10 | $0.40 | ~$0.16 | Fallback if cost ever matters; weaker on 8-field extraction |
| `gpt-4o-mini` | $0.15 | $0.60 | ~$0.24 | Current `.env.example` default; older, less reliable at strict schemas |
| `gpt-5-mini` | $0.25 | $2.00 | ~$0.60 | **Avoid — see below** |

*(Prices approximate; confirm at platform.openai.com/pricing. Assumes ~600 input / ~250 output tokens
per report.)*

**Why not a gpt-5 reasoning model:** `build_prompt_chain()` hardcodes `temperature=0.0`
([prompt_builder.py:123](pipeline/prompt_builder.py#L123)). The gpt-5 reasoning family rejects
`temperature`, so picking one turns a zero-code config change into a code change plus a new
`reasoning_effort` branch — for a task that is mechanical field extraction from a two-paragraph report
and needs no reasoning tokens at all. `gpt-4.1-mini` costs the same and just works.

At ~$0.65 for the entire 1000-report dataset, do not optimise this further. The GPU time dominates by
three orders of magnitude.

### 1.2 SD3 modes: build both

| | `SD3_MODE=txt2img` (default) | `SD3_MODE=img2img` |
|---|---|---|
| Diffusers class | `StableDiffusion3Pipeline` | `StableDiffusion3Img2ImgPipeline` |
| Reference X-ray | ignored, warned once | `image=` init latent + `strength` knob |
| Geometry from | explicit `height`/`width` (bucket table) | the resized init image |
| Output | pure synthetic, no real-pixel provenance | conditioned on a real X-ray |
| Use for | detector test sets, unconditional generation | anatomy/body-habitus fidelity to a patient |

They share one module, one weight load (`from_pipe`, Step 3a), one prompt adapter, one retry wrapper. The
only divergence is ~30 lines of call kwargs.

**Caveat, recorded not acted on:** img2img output derives from real pixels, so if these become the
"synthetic" half of a real-vs-synthetic detector benchmark, that half is contaminated — the detector may
learn the residue of the real source rather than the generator's fingerprint. That is a reason to choose
`txt2img` *for that experiment*, not to withhold the mode. Hence txt2img is the default, and
`metadata.json` records the effective mode per entry so consumers can tell the halves apart.

### 1.3 Prompt condensing: rule-based

Build the CLIP prompt from the structured fields already on `StructuredRadiologyPrompt`. Deterministic,
free, no second API call, no new failure mode. The fields exist precisely so this is mechanical.

### 1.4 Two-session workflow: yes (`--prompts-from`)

Extraction runs on CPU with the OpenAI key; the GPU session only diffuses. Saves ~30 min of Kaggle GPU
quota per 1000-report run and makes prompt iteration free. Step 8 — optional, do it if iterating.

---

## 2. What SD3 changes vs. the API back-ends

| Assumption in current code | DALL·E / Gemini | SD3 | Consequence |
|---|---|---|---|
| Client built per call inside `generate_image_*` | fine (cheap HTTP client) | **~15 GB download + 30–90 s to GPU** | Module-level cached singleton, preloaded before the loop |
| Prompt can be arbitrarily long | yes | **CLIP-L/G truncate at 77 tokens**; T5 at 256 (512 max) | Dedicated prompt adapter — see below |
| Aspect ratio is a string (`"3:4"`) | yes | takes explicit `height`/`width`, multiples of 16 | Dimension bucket solver, not a ratio picker |
| Reference images pass through | Gemini only | **img2img mode only** | Two pipeline classes sharing one weight load |
| Runs anywhere | yes | needs a 16–24 GB GPU + accepted HF license | Kaggle notebook + HF token |

**The prompt-truncation problem — the one that will silently ruin output quality.**
`format_image_prompt()` emits ~**360 words of static boilerplate** plus findings, roughly **400–500 CLIP
tokens**. SD3's CLIP encoders see only the **first 77**, which for a report with reference images is the
`[REFERENCE IMAGES]` header and modality boilerplate — the clinical findings never reach CLIP at all.

`StableDiffusion3Pipeline` accepts `prompt` / `prompt_2` / `prompt_3` as three separate strings routed to
CLIP-L / CLIP-G / T5. That is the escape hatch, and the reason `sd3_prompt_adapter.py` exists.

---

## 3. Files touched

```text
NEW  pipeline/sd3_prompt_adapter.py     # long prompt → (clip_prompt, t5_prompt) + negative prompt
NEW  pipeline/sd3_generator.py          # cached model + txt2img/img2img
NEW  requirements-sd3.txt               # torch/diffusers stack, kept out of base requirements
NEW  test_sd3.py                        # smoke checks — no GPU, no download
NEW  sd3-kaggle-notebook.ipynb          # the cloud runner
EDIT pipeline/config.py                 # SD3_* settings + SD3_SUPPORTED_DIMENSIONS table
EDIT pipeline/image_generator.py        # one dispatcher branch → sd3
EDIT pipeline/pipeline.py               # --generator choices, SD3 flags, --output-subdir, preload
EDIT .env.example                       # OpenAI defaults + every SD3_* variable
EDIT requirements.txt                   # comment pointing at requirements-sd3.txt
EDIT ARCHITECTURE.md                    # §2 tree, §4 table, §5 flow, §6 config, §8 rules
EDIT README.md                          # SD3 quickstart, both modes, OpenAI setup
```

Unchanged: `models.py`, `report_parser.py`, `view_splitter.py`, `image_prompt_formatter.py`,
**`prompt_builder.py`** (the OpenAI path already works). The SD3 back-end consumes the exact
`(prompt, uid, view_suffix, source_dimensions, image_paths)` contract the dispatcher already passes.

---

## 4. Implementation steps

### Step 1 — `pipeline/sd3_prompt_adapter.py`

Pure functions, no I/O, no SDK imports. House style per ARCHITECTURE.md §7 (`view_splitter.py` is the
template).

```python
def split_prompt_for_sd3(prompt_text: str,
                         sp: StructuredRadiologyPrompt | None = None) -> tuple[str, str]:
    """Return (clip_prompt, t5_prompt) for SD3's dual text-encoder stack."""

def build_negative_prompt(extra: str | None = None) -> str:
    """Diffusion-style comma-separated negatives."""
```

- **`t5_prompt`** = full `format_image_prompt()` output minus the `[REFERENCE IMAGES]` and
  `[NEGATIVE CONSTRAINTS — STRICT]` sections.
- **`clip_prompt`** = from structured fields when `sp` is given:
  `"{modality}, {view} view, {anatomical_region}, {first findings clause}, grayscale diagnostic radiograph"`.
  Falls back to regex-extracting the `[MODALITY CONDITIONING]` + `[TEXTUAL PROMPTING…]` bodies when only
  text is available. Hard-capped by `_truncate_to_words(..., 60)` — conservative stand-in for 77 tokens.
- **`build_negative_prompt`** converts the `[NEGATIVE CONSTRAINTS]` prose into the noun-phrase form
  diffusion negatives actually want: `"text, letters, words, numbers, labels, captions, annotations,
  watermark, overlay, UI elements, border, frame, color, photograph, illustration, cartoon, blurry,
  low quality, deformed anatomy"`, plus `config.SD3_NEGATIVE_PROMPT` when set. **Moving the
  prohibitions out of the positive prompt matters** — "do NOT render text" inside a positive prompt
  reliably summons text in diffusion models.
- Private helpers: `_strip_section`, `_extract_section`, `_truncate_to_words`.

### Step 2 — `pipeline/config.py`

```python
HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN", "")
SD3_MODEL_ID: str      = os.getenv("SD3_MODEL_ID", "stabilityai/stable-diffusion-3-medium-diffusers")
SD3_MODE: str          = os.getenv("SD3_MODE", "txt2img")            # txt2img | img2img
SD3_STEPS: int         = int(os.getenv("SD3_STEPS", "28"))            # Stability default
SD3_GUIDANCE_SCALE: float   = float(os.getenv("SD3_GUIDANCE_SCALE", "7.0"))
SD3_HEIGHT: int        = int(os.getenv("SD3_HEIGHT", "1024"))
SD3_WIDTH: int         = int(os.getenv("SD3_WIDTH", "1024"))
SD3_DTYPE: str         = os.getenv("SD3_DTYPE", "float16")            # float16 | bfloat16
SD3_NEGATIVE_PROMPT: str    = os.getenv("SD3_NEGATIVE_PROMPT", "")
SD3_SEED: int | None   = int(v) if (v := os.getenv("SD3_SEED", "")) else None
SD3_MAX_SEQUENCE_LENGTH: int = int(os.getenv("SD3_MAX_SEQUENCE_LENGTH", "512"))
SD3_ENABLE_CPU_OFFLOAD: bool = os.getenv("SD3_ENABLE_CPU_OFFLOAD", "true").lower() == "true"
SD3_ENABLE_VAE_SLICING: bool = os.getenv("SD3_ENABLE_VAE_SLICING", "true").lower() == "true"
SD3_DROP_T5: bool      = os.getenv("SD3_DROP_T5", "false").lower() == "true"
SD3_IMG2IMG_STRENGTH: float = float(os.getenv("SD3_IMG2IMG_STRENGTH", "0.75"))  # img2img only
```

Plus the dimension table (ARCHITECTURE.md §8: extend the tables, never hardcode geometry):

```python
# SD3 native buckets: ~1 MP each, every side a multiple of 16.
SD3_SUPPORTED_DIMENSIONS: dict[str, tuple[int, int]] = {
    "1:1":  (1024, 1024),
    "4:3":  (1152,  896),   "3:4":  ( 896, 1152),
    "3:2":  (1216,  832),   "2:3":  ( 832, 1216),
    "16:9": (1344,  768),   "9:16": ( 768, 1344),
    "5:4":  (1088,  896),   "4:5":  ( 896, 1088),
}
```

Chest X-rays are overwhelmingly portrait, so `3:4` / `4:5` / `2:3` are the buckets that will get used.

Also change the `CHAT_MODEL` default from `gemini-2.0-flash` → `gpt-4.1-mini`, and `IMAGE_GENERATOR`
default to `sd3`.

### Step 3 — `pipeline/sd3_generator.py`

Both modes in one pass. Retrofitting img2img later would mean rewriting the cache key and metadata
contract anyway.

**3a — Cached pipeline singleton**

```python
_PIPELINE_CACHE: dict[str, object] = {}

def load_sd3_pipeline(mode: str | None = None):
    """Load (and cache) the SD3 pipeline for a mode. Idempotent; safe to call per image."""
```

- Deferred imports (`import torch`, `from diffusers import ...`) **inside** the function — mandatory per
  ARCHITECTURE.md §8, so the package still imports without torch and `test_pipeline.py` keeps passing.
- **Keyed on `mode` so both pipelines coexist.** txt2img loads from the hub; img2img is then built with
  `StableDiffusion3Img2ImgPipeline.from_pipe(txt2img_pipe)`, which rebinds **the same transformer, VAE,
  and three text encoders** into the other class — no second download, no second VRAM copy. Without
  this, a mixed run needs ~30 GB and a second 15 GB pull. This is what makes both modes nearly free, and
  why `load_sd3_pipeline("img2img")` must route through the txt2img loader rather than calling
  `from_pretrained` twice.
- `from_pretrained(..., torch_dtype=<resolved>, variant="fp16", token=config.HUGGINGFACE_TOKEN or None)`.
  `variant="fp16"` halves the download.
- `SD3_DROP_T5=true` → `text_encoder_3=None, tokenizer_3=None`, freeing ~10 GB but reducing the model to
  the two 77-token CLIP encoders. Escape hatch for a 16 GB card; **it discards the long-prompt path this
  design is built around**, so warn loudly when it is on.
- Device: `cuda` → `mps` → `cpu`. `SD3_ENABLE_CPU_OFFLOAD` → `pipe.enable_model_cpu_offload()`, which is
  **mutually exclusive with `.to("cuda")`** — offload manages placement itself, so branch, never both.
  `enable_vae_slicing()` when configured.
- dtype guard: fp16 on T4 (no bf16 tensor cores), bf16 on L4/A100/4090. Warn if bf16 is requested on a
  device with compute capability < 8.0 — this is the classic all-black-output bug.

**3b — Dimension solver**

```python
def compute_sd3_dimensions(width: int, height: int) -> tuple[int, int]:
```
Reuses `image_generator.compute_best_aspect_ratio(w, h, ratios=...)` over a `{key: w/h}` view of
`SD3_SUPPORTED_DIMENSIONS` — one ratio-matching implementation, two tables.

**3c — Mode resolution + init image**

```python
def _resolve_mode(mode: str | None, image_paths: list[Path] | None) -> str:
    """Effective mode, downgrading img2img → txt2img when no reference image exists."""

def _load_init_image(path: Path, size: tuple[int, int]) -> "PIL.Image.Image":
    """Reference X-ray → SD3-ready RGB init image at (w, h)."""
```

- **txt2img with reference images present:** warn once per record —
  `"⚠ SD3 txt2img ignores the N reference image(s); set SD3_MODE=img2img to use them."` Same shape as
  the DALL·E warning, so metadata consumers see no surprises.
- **img2img with no reference image — the case that will silently produce the wrong dataset.**
  `run_pipeline` already passes `image_paths=None` when a projection file is missing on disk. Downgrade
  that record to txt2img and warn; **never raise**. A 1000-record batch must not die on record 12
  because one PNG is absent. Record the *effective* mode in metadata.
- **Init image:** `open()` → `.convert("RGB")` (X-rays are 8-bit greyscale; the VAE needs 3 channels) →
  `.resize((w, h), LANCZOS)`. Resize to the **bucket** dimensions, not the raw source — Indiana X-rays
  are ~2000×2500, far above SD3's 1 MP sweet spot; passing them unresized is both an OOM risk and a
  quality regression.
- Multi-view records may carry both Frontal and Lateral. Use `image_paths[0]` — the orchestrator already
  reads dimensions from that same entry, so geometry stays consistent. Matching the init image to
  `view_suffix` is a future refinement; noted, not built.

**3d — Generation**

```python
def generate_image_sd3(prompt, uid, image_paths=None, view_suffix=None,
                       source_dimensions=None, max_retries=3,
                       structured_prompt=None, mode=None) -> Path:
```

Steps 1–4 and 7–9 are shared by both modes; only 5–6 branch.

1. Resolve `(h, w)` from `source_dimensions` via the bucket table, else `SD3_HEIGHT`/`SD3_WIDTH`.
2. `clip_prompt, t5_prompt = split_prompt_for_sd3(prompt, structured_prompt)`;
   `negative = build_negative_prompt(config.SD3_NEGATIVE_PROMPT)`.
3. `mode = _resolve_mode(mode, image_paths)`.
4. `pipe = load_sd3_pipeline(mode)` (cached).
5. Seed: `torch.Generator(device).manual_seed(config.SD3_SEED + uid)` when `SD3_SEED` is set — runs
   reproducible, every uid still distinct. Both modes take the same generator, so a fixed seed makes a
   txt2img/img2img pair directly comparable.
6. Branch:
   ```python
   common = dict(
       prompt=clip_prompt, prompt_2=clip_prompt, prompt_3=t5_prompt,
       negative_prompt=negative, negative_prompt_2=negative, negative_prompt_3=negative,
       num_inference_steps=config.SD3_STEPS,
       guidance_scale=config.SD3_GUIDANCE_SCALE,
       max_sequence_length=config.SD3_MAX_SEQUENCE_LENGTH,
       generator=generator,
   )

   if mode == "txt2img":
       image = pipe(**common, height=h, width=w).images[0]
   else:
       init = _load_init_image(image_paths[0], (w, h))
       image = pipe(**common, image=init, strength=config.SD3_IMG2IMG_STRENGTH).images[0]
   ```
   `height`/`width` are **not valid** on the img2img class — geometry comes from the init image, which is
   why `_load_init_image` resizes to the same bucket. Note img2img runs only
   `strength × num_inference_steps` denoising steps, so at `strength=0.75` a 28-step request costs ~21 —
   img2img is meaningfully *faster* per image.
7. Save to `config.IMAGES_DIR / f"{uid}_{view_suffix}.png"` — filename contract unchanged. **Both modes
   write the same filename**, hence `--output-subdir` in Step 4.
8. Retry 3× with `2**attempt` backoff. On `torch.cuda.OutOfMemoryError`, `torch.cuda.empty_cache()`
   before retrying — the one failure mode where a retry can genuinely succeed. Raise `RuntimeError`
   after the last attempt so `run_pipeline` records `error_image` and the batch continues.
9. `print()` the same `✓ Image saved: …` shape as the other back-ends, with `mode=` and (img2img)
   `strength=` in the parenthetical.

**`strength` semantics** — the knob that will be tuned: `0.0` returns the input untouched, `1.0` ignores
it (≡ txt2img). Useful band for "keep this patient's anatomy, restyle the pathology" is ~0.5–0.8. Below
~0.4 the output is a lightly filtered copy of a real X-ray, which is neither synthetic nor safe to
redistribute as such.

### Step 4 — wiring

**`image_generator.py`** — one branch, delegation only:
```python
elif gen == "sd3":
    from .sd3_generator import generate_image_sd3
    return generate_image_sd3(prompt, uid, image_paths=image_paths,
                              view_suffix=view_suffix,
                              source_dimensions=source_dimensions)
```
Update the `ValueError` message to list all three names.

**`pipeline.py`**
- `choices=["dalle", "gemini", "sd3"]` on `--generator`.
- New flags, all defaulting to `None` and overwriting the matching `config.*` before the loop (the
  established `value = arg or config.DEFAULT` pattern, kept optional so existing invocations and the
  current Kaggle notebook keep working):
  `--sd3-mode {txt2img,img2img}`, `--sd3-strength`, `--sd3-steps`, `--sd3-guidance`, `--sd3-seed`,
  `--sd3-model-id`. Warn (don't error) if `--sd3-strength` is passed with `txt2img`.
- **`--output-subdir NAME`** — `IMAGES_DIR` becomes `output/images/<NAME>/`, metadata goes to
  `output/metadata_<NAME>.json`. This is what lets txt2img and img2img runs of the same uids coexist
  instead of overwriting (Step 3d, item 7), and it makes parameter sweeps possible.
- **Validate img2img at startup:** if effective mode is `img2img` and `--images-dir` is absent, print a
  prominent warning once (`"SD3_MODE=img2img but no --images-dir — every record falls back to txt2img"`)
  rather than emitting the per-record warning 1000 times. Fail fast on the config mistake, degrade
  gracefully on the per-record one.
- **Preload before the loop:** when `gen == "sd3"` and images aren't skipped, call
  `load_sd3_pipeline(mode)` *before* `tqdm` starts, so the ~2 min load isn't charged to the first report
  and the ETA is honest — the SD3 analogue of the "chain is built once" rule. In img2img mode preload
  **both** entries (txt2img, then `from_pipe`) so a mid-run downgrade never stalls on a cold load.
- Metadata: `entry["generator"] = gen`, and for SD3
  `entry["sd3_params"] = {mode, steps, guidance, seed, dimensions, model_id}` plus `strength` and
  `init_image` in img2img. `mode` is the **effective** mode after any downgrade. This is what makes a
  run reproducible and lets a consumer separate pure-synthetic entries from reference-conditioned ones.

### Step 5 — `test_sd3.py`

Plain script, `test_aspect_ratio.py` style. **No GPU, no download.**

1. `(w, h) → expected bucket` table through `compute_sd3_dimensions`, including the portrait shapes
   actually present in the Indiana set.
2. `split_prompt_for_sd3` invariants: `len(clip_prompt.split()) <= 60`; clinical findings present in
   `clip_prompt` (not just boilerplate); `[NEGATIVE CONSTRAINTS` absent from `t5_prompt`;
   `[TEXTUAL PROMPTING` present in `t5_prompt`.
3. `import pipeline.sd3_generator` succeeds **without torch installed** — proves the deferred-import rule
   is honoured. This is the gate that keeps the package usable for non-SD3 users.
4. `generate_image(..., generator="sd3")` dispatch reaches the SD3 path (monkeypatched).
5. **Mode-resolution table:** `("img2img", None) → "txt2img"`, `("img2img", [p]) → "img2img"`,
   `("txt2img", [p]) → "txt2img"`, `(None, …) → config default`, unknown string → `ValueError`. Pure
   logic, no GPU cost, and it guards the edge case most likely to silently produce the wrong dataset.
6. `_load_init_image` on a temp greyscale PNG returns RGB mode at exactly the requested `(w, h)`.

### Step 6 — supporting files

- **`requirements-sd3.txt`** — `torch>=2.1`, `diffusers>=0.31`, `transformers>=4.44`, `accelerate>=0.33`,
  `sentencepiece`, `protobuf`, `safetensors`. Kept **out of `requirements.txt`** so a CPU-only clone
  doesn't pull a multi-GB torch wheel; base file gets a comment pointing here.
- **`.env.example`** — `OPENAI_API_KEY`, `CHAT_MODEL=gpt-4.1-mini`, `IMAGE_GENERATOR=sd3`,
  `HUGGINGFACE_TOKEN`, every `SD3_*` variable with a comment. Note that the Google/Vertex block is now
  optional and only needed for the `gemini` back-end.
- **`ARCHITECTURE.md`** — §2 tree, §4 module table (two rows), §5 flow (SD3 path), §6 config, and two
  new §8 rules: *"SD3 loads a multi-GB model — cache it in a module-level singleton and preload before
  the record loop; never construct it per image"* and *"SD3's CLIP encoders truncate at 77 tokens;
  always route long prompts through `sd3_prompt_adapter`."*
- **`README.md`** — "SD3 (local GPU)" quickstart: HF license → token → `pip install -r
  requirements-sd3.txt` → both invocations, plus a short "which mode do I want?" paragraph reusing the
  §1.2 table. That is the first question any user of this back-end will have.

### Step 7 — `sd3-kaggle-notebook.ipynb`

**Kaggle UI settings the user must set: Accelerator = GPU L4 ×4 (preferred) or T4 ×2, Internet = ON.**

| # | Type | Contents |
|---|---|---|
| 0 | md | Prerequisites: HF license accepted at `huggingface.co/stabilityai/stable-diffusion-3-medium`; Kaggle secrets `HF_TOKEN` and `OPENAI_API_KEY`; accelerator + internet on. **No GCP service account needed.** |
| 1 | code | `!nvidia-smi` + `torch.cuda.get_device_properties(0)` → name, VRAM, compute capability. **Assert a GPU is present** — the most common way this notebook wastes an hour. |
| 2 | code | `!git clone https://github.com/whiteflags26/synth-dataset-pipeline.git` ; `%cd synth-dataset-pipeline` |
| 3 | code | `!pip install -q -U diffusers transformers accelerate sentencepiece protobuf langchain langchain-openai openai pandas python-dotenv Pillow pydantic tqdm`. **Do not reinstall torch** — Kaggle ships it, and a 3 GB reinstall frequently breaks CUDA. **No `langchain-google-*` or `google-cloud-aiplatform`** — OpenAI + SD3 needs neither, which cuts install time substantially. Print `diffusers.__version__`. |
| 4 | code | `os.environ["HF_HOME"] = "/kaggle/working/hf_cache"` **before any HF import** — the default cache is on a small volume and a ~15 GB pull will exhaust it. Read `HF_TOKEN` from `UserSecretsClient`, `huggingface_hub.login(...)`. |
| 5 | code | Write `.env` from Kaggle secrets: `OPENAI_API_KEY`, `CHAT_MODEL=gpt-4.1-mini`, `HUGGINGFACE_TOKEN`, `IMAGE_GENERATOR=sd3`, `SD3_*` with VRAM-aware defaults (`float16` + offload when VRAM < 20 GB, else `bfloat16`, no offload). A single `MODE = "txt2img"` variable at the top drives `SD3_MODE` and the run cells below. Never hardcode a key in a cell. |
| 6 | code | Dataset sanity check — `images_dir` exists, file count, sample filenames. Catches a stale Kaggle dataset path *before* the model downloads. |
| 7 | code | **Warm-up:** `load_sd3_pipeline("txt2img")` + one 4-step throwaway, then `load_sd3_pipeline("img2img")` + one 4-step img2img on a real X-ray. Proves both classes construct and that `from_pipe` shares weights — print `torch.cuda.memory_allocated()` before and after the second load; it should barely move. Isolates download/OOM/gated-repo failures into a cell that fails in 3 minutes instead of 40. |
| 8 | code | **Prompt smoke test:** one report through `extract_structured_prompt` with `gpt-4.1-mini`, printing the structured fields and the resulting `clip_prompt` / `t5_prompt`. Confirms the OpenAI key works and the CLIP prompt actually contains findings — cheap, and it catches a bad key before a long run. |
| 9a | code | **txt2img run** — `!python -m pipeline --csv indiana_reports.csv --projections-csv indiana_projections.csv --images-dir <path> --generator sd3 --sd3-mode txt2img --output-subdir txt2img --limit 10` |
| 9b | code | **img2img run** — same with `--sd3-mode img2img --sd3-strength 0.75 --output-subdir img2img --limit 10`. Same uids, same seed → directly comparable. |
| 10 | code | **Three-column grid:** original X-ray │ txt2img │ img2img, one row per uid, from the two metadata files. Makes the mode choice empirical rather than an argument. |
| 11 | code | **Strength sweep:** one uid, `strength ∈ {0.4, 0.55, 0.7, 0.85, 1.0}`, filmstrip with the original at left. Calls `generate_image_sd3` in-process — pipeline already loaded, no CLI, no reload. Pins the useful band for *this* dataset instead of taking 3d on faith. |
| 12 | code | Export: walk both output subdirs → `verification_data/{txt2img,img2img}/` + zip + `FileLink`. Adapted from cell 8 of the existing notebook. |
| 13 | md | Troubleshooting: gated-repo 401, CUDA OOM, `no space left on device`, bf16-on-T4 black images, OpenAI 401/429, and "img2img silently produced txt2img output" → missing `--images-dir` or absent PNG. |

**Kaggle constraints:** 12 h max session, 30 h/week GPU quota. `/kaggle/working` is ~20 GB and is where
the ~15 GB fp16 model lands — committing the notebook with the model there will fail the save, so the
export cell zips only `output/`, and cell 4 points `HF_HOME` somewhere the commit ignores. Budget
**8–15 s/image** on an L4 at 28 steps; **60–120 s/image** on a T4 with CPU offload. A 1000-image run is
an L4 job, not a T4 job.

### Step 8 — `--prompts-from` (optional; do it if iterating)

Add `--prompts-from <path>` to `run_pipeline`: skip `load_reports` + `build_prompt_chain`, read the
existing metadata JSON, and call `generate_image` for each `entry["views"][*]["image_prompt"]`. No LLM
credentials, no CSV, no chain — the GPU session only diffuses.

```bash
python -m pipeline --limit 200 --skip-images                                 # CPU, OpenAI key only
python -m pipeline --prompts-from output/metadata.json --generator sd3       # GPU only
```

Make `--prompts-from` and `--csv` mutually exclusive, and write to a distinct output subdir so the
source prompt file is never clobbered mid-run.

---

## 5. Runbook

**Local setup**
```bash
pip install -r requirements.txt -r requirements-sd3.txt
cp .env.example .env      # set OPENAI_API_KEY and HUGGINGFACE_TOKEN
python test_pipeline.py && python test_aspect_ratio.py && python test_sd3.py
```

**Prompt-only check (no GPU, costs ~$0.002)**
```bash
python -m pipeline --limit 3 --skip-images
```

**txt2img**
```bash
python -m pipeline --generator sd3 --sd3-mode txt2img \
  --csv indiana_reports.csv --limit 5 --output-subdir txt2img
```

**img2img**
```bash
python -m pipeline --generator sd3 --sd3-mode img2img --sd3-strength 0.75 \
  --csv indiana_reports.csv \
  --projections-csv indiana_projections.csv \
  --images-dir <path-to-images_normalized> \
  --limit 5 --output-subdir img2img
```

---

## 6. Order of execution

1. `sd3_prompt_adapter.py` — pure, zero dependencies, testable immediately.
2. `config.py` + `.env.example` (SD3 settings, OpenAI defaults).
3. `sd3_generator.py` — **both modes together**.
4. Wiring: dispatcher branch, CLI flags, `--output-subdir`, preload, metadata.
5. `test_sd3.py` green locally **without torch installed**.
6. `requirements-sd3.txt`, README, ARCHITECTURE. Commit + push (the notebook clones from GitHub, so this
   must land before Step 7 can run).
7. `sd3-kaggle-notebook.ipynb`; first GPU run at `--limit 2`, both modes.
8. Tune against real output: `SD3_STEPS` / `SD3_GUIDANCE_SCALE` / negative prompt for txt2img, strength
   sweep for img2img. **Expect the CLIP prompt wording to need a pass or two** — that is where output
   quality is won or lost, and it affects both modes equally since they share the adapter.
9. `--prompts-from`, if the two-session workflow proves worth it.

---

## 7. Definition of done

- [ ] `python -m pipeline --limit 3 --skip-images` extracts structured prompts via `gpt-4.1-mini` with no
      Google credentials present anywhere.
- [ ] `--sd3-mode txt2img --limit 5` produces 5 PNGs at the correct per-uid aspect ratio.
- [ ] `--sd3-mode img2img --images-dir … --limit 5` produces 5 PNGs visibly conditioned on their
      reference X-rays; uids **without** a reference PNG fall back to txt2img with a warning instead of
      failing the batch.
- [ ] `metadata.json` records the effective `sd3_params.mode` (plus `strength`, `init_image` in img2img)
      for every entry.
- [ ] `test_pipeline.py`, `test_aspect_ratio.py`, `test_sd3.py` all pass; `test_sd3.py` passes in an env
      **without torch**.
- [ ] `dalle` and `gemini` back-ends still work unchanged.
- [ ] Kaggle notebook runs top to bottom on a fresh session; cell 10 shows original / txt2img / img2img
      side by side.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| Gated HF repo → 401 | Cell 0 prerequisites; cell 7 warm-up fails fast with a clear message |
| CUDA OOM on a 16 GB T4 | `SD3_ENABLE_CPU_OFFLOAD` default on, `SD3_DROP_T5` escape hatch, `empty_cache()` on OOM retry |
| Kaggle disk exhausted by the ~15 GB model | `HF_HOME` → `/kaggle/working/hf_cache`, `variant="fp16"` |
| Two pipeline classes double VRAM if loaded naively | `from_pipe` shares weights; warm-up cell asserts it by printing allocated VRAM before/after |
| SD3 renders text overlays anyway (it is unusually good at text — a real weakness here) | Prohibitions moved into `negative_prompt` as noun phrases; may need guidance tuning |
| **img2img silently degrading to txt2img** across a whole run (missing `--images-dir`, unattached dataset) — yields a "reference-conditioned" dataset that isn't one | Startup validation on the config mistake; effective mode recorded per entry; cell 10's grid makes it visible immediately |
| **img2img at low strength yields near-copies of real X-rays** — neither synthetic nor neutral to redistribute | Default `0.75`; strength sweep pins the floor empirically; do not ship a set generated below ~0.5 without inspecting it |
| Medical fidelity: SD3 has little chest-radiograph training data | Out of scope to fix — flag it. txt2img output looks plausible to a layperson, not a radiologist. **img2img is the main lever available**; beyond that it's a fine-tune, not prompt tuning |
| `diffusers` API drift on `prompt_3` / `max_sequence_length` | Pin `diffusers>=0.31`, print the version in cell 3, verify signatures against the installed build before writing Step 3d |
| OpenAI structured-output schema rejection | `with_structured_output` defaults to function-calling and handles `Optional` fields fine; if adherence is poor, switch to `method="json_schema", strict=True` |

---

## 9. Verify at implementation time

- **Kaggle dataset path.** The existing notebook uses
  `/kaggle/input/datasets/raddar/chest-xrays-indiana-university/images/images_normalized`; path shapes
  vary by how the dataset is attached. Cell 6 exists to catch this.
- **`diffusers` accepts `prompt_3` with `max_sequence_length=512`** on the installed build — check with
  `inspect.signature` in cell 3 rather than trusting the docs.
- **`gpt-4.1-mini` pricing and availability** on the user's account tier.
- **Checkpoint choice.** `stabilityai/stable-diffusion-3.5-medium` is newer and less gated, but
  `sd3_img_generation_config.md` §1 argues for plain SD3 Medium to match the SPAI paper's "SD3". Keep
  `stable-diffusion-3-medium-diffusers` as the default; `SD3_MODEL_ID` makes switching a one-line env
  change.

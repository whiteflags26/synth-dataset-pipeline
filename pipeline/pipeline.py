"""
Pipeline Orchestrator — end-to-end flow:

  CSV  →  Report Parser  →  Structured Prompt (GPT/Gemini)  →  Image Prompt
       →  DALL-E / Gemini / Stable Diffusion 3
                         ↑
              Optional: indiana_projections.csv + images_dir
              → reference images passed as multimodal input (Gemini)
                or as the img2img init image (SD3)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image as PILImage
from tqdm import tqdm

from . import config
from .report_parser import load_projections, load_reports
from .prompt_builder import build_prompt_chain, extract_structured_prompt
from .image_prompt_formatter import format_image_prompt
from .image_generator import compute_best_aspect_ratio, generate_image
from .view_splitter import split_prompt_by_views


def _apply_output_subdir(name: str) -> None:
    """
    Redirect output into a named subdirectory.

    Lets several runs of the same uids coexist — txt2img beside img2img, or a
    parameter sweep — instead of overwriting each other, since the filename
    contract is <uid>.png regardless of backend or settings.
    """
    config.IMAGES_DIR = config.OUTPUT_DIR / "images" / name
    config.METADATA_FILE = config.OUTPUT_DIR / f"metadata_{name}.json"
    config.IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def _apply_shard(
    uids: list[int],
    num_shards: int | None,
    shard_index: int | None,
) -> set[int] | None:
    """
    Validate a shard request and return the set of uids assigned to it.

    Assignment is uid % num_shards == shard_index rather than a slice of the
    list, so it stays stable if the CSV is re-ordered or filtered elsewhere,
    and two shards run from the same --csv never need to agree on ordering.
    Returns None when no sharding was requested (num_shards is None).
    """
    if num_shards is None:
        return None
    if shard_index is None:
        raise ValueError("--shard-index is required when --num-shards is given")
    if num_shards < 1 or not (0 <= shard_index < num_shards):
        raise ValueError(f"--shard-index must be in [0, {num_shards})")
    return {uid for uid in uids if uid % num_shards == shard_index}


def _load_completed_uids(metadata_path: Path) -> tuple[dict[int, dict], set[int]]:
    """
    Read an existing metadata file and determine which uids are fully done.

    A uid counts as done when prompt extraction succeeded and every view was
    successfully generated (no error_prompt, no error_image, every view has an
    image_path). Anything else — a failed extraction, a partially generated
    multi-view report — is retried on resume rather than patched in place;
    redoing a failed record is cheap next to the complexity of splicing in
    just the missing view.

    Returns
    -------
    (by_uid, completed_uids)
        by_uid         : every existing entry, keyed by uid, so completed ones
                         can be carried forward into a resumed run's results.
        completed_uids : the subset that should be skipped on resume.
    """
    if not Path(metadata_path).exists():
        return {}, set()

    with open(metadata_path, encoding="utf-8") as f:
        existing = json.load(f)

    by_uid = {e["uid"]: e for e in existing}
    completed = {
        uid
        for uid, entry in by_uid.items()
        if not entry.get("error_prompt")
        and entry.get("views")
        and all("image_path" in v for v in entry["views"])
    }
    return by_uid, completed


def _save_metadata(results: list[dict]) -> None:
    """
    Write results to config.METADATA_FILE.

    Called after every record, not just once at the end, so a killed session
    (Kaggle's 12h cap, an OOM, a manual stop) loses at most the record that
    was in flight. --resume reads this same file back on the next invocation.
    """
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def _sd3_params(
    sd3_mode: str | None,
    ref_image_paths: list[Path] | None,
    source_dimensions: tuple[int, int] | None,
) -> dict:
    """
    Record the SD3 settings that actually produced an image.

    The mode stored here is the *effective* one after any img2img → txt2img
    downgrade, so metadata reflects what ran rather than what was requested.
    Without this a mixed run is indistinguishable from a pure one afterwards.
    """
    from .sd3_generator import _resolve_mode, compute_sd3_dimensions

    effective = _resolve_mode(sd3_mode, ref_image_paths)
    width, height = (
        compute_sd3_dimensions(*source_dimensions)
        if source_dimensions
        else (config.SD3_WIDTH, config.SD3_HEIGHT)
    )

    params: dict = {
        "mode": effective,
        "model_id": config.SD3_MODEL_ID,
        "steps": config.SD3_STEPS,
        "guidance_scale": config.SD3_GUIDANCE_SCALE,
        "seed": config.SD3_SEED,
        "width": width,
        "height": height,
    }
    if effective == "img2img" and ref_image_paths:
        params["strength"] = config.SD3_IMG2IMG_STRENGTH
        params["init_image"] = ref_image_paths[0].name

    return params


def run_pipeline(
    csv_path: str | Path,
    limit: int | None = None,
    offset: int | None = None,
    generator: str | None = None,
    skip_image_generation: bool = False,
    projections_csv: str | Path | None = None,
    images_dir: str | Path | None = None,
    sd3_mode: str | None = None,
    output_subdir: str | None = None,
    resume: bool = False,
    num_shards: int | None = None,
    shard_index: int | None = None,
) -> list[dict]:
    """
    Execute the full pipeline.

    Parameters
    ----------
    csv_path              : path to indiana_reports.csv
    limit                 : process only the first N reports (after offset)
    offset                : skip the first N reports
    generator             : "dalle", "gemini", or "sd3" (overrides config)
    skip_image_generation : if True, stop after prompt generation (useful for testing)
    projections_csv       : optional path to indiana_projections.csv.
                            Required when images_dir is provided.
    images_dir            : optional path to the directory containing the
                            original X-ray PNG images (e.g. on Kaggle:
                            /kaggle/input/.../images_normalized/).
                            When provided along with projections_csv,
                            reference images are passed as multimodal input.
    sd3_mode              : "txt2img" or "img2img" (SD3 generator only)
    output_subdir         : write images to output/images/<name>/ and metadata
                            to output/metadata_<name>.json instead of the
                            defaults, so parallel runs do not overwrite.
    resume                : skip uids that already have a complete, error-free
                            entry in the target metadata file (see
                            output_subdir), and continue appending into it.
                            Failed or partial entries are retried. Combined
                            with the fact that metadata is saved after every
                            record, an interrupted run only needs the same
                            command re-run with --resume to pick back up.
    num_shards, shard_index : split the loaded reports across N processes by
                            uid % num_shards == shard_index, so two (or more)
                            processes — typically one per GPU on a multi-GPU
                            Kaggle session — can each own a disjoint slice of
                            the work. Pair with a distinct --output-subdir per
                            shard so their metadata/images never collide, then
                            merge_shards.py combines the results afterward.

    Returns
    -------
    list of metadata dicts for each processed report — the full accumulated
    set when resume=True, including entries carried over unchanged.
    """
    gen = generator or config.IMAGE_GENERATOR
    images_dir_path = Path(images_dir) if images_dir else None

    if output_subdir:
        _apply_output_subdir(output_subdir)

    # ── SD3 configuration checks ─────────────────────────────────────────
    effective_sd3_mode = (sd3_mode or config.SD3_MODE).lower() if gen == "sd3" else None
    if effective_sd3_mode == "img2img" and not images_dir_path:
        print(
            "⚠  SD3_MODE=img2img but no --images-dir was given — every record "
            "will fall back to txt2img. Pass --images-dir and --projections-csv "
            "to condition on the real X-rays.\n"
        )

    # ── Load projections lookup (optional) ───────────────────────────────
    projections = None
    if projections_csv and images_dir_path:
        print(f"🗂  Loading projections from {projections_csv} ...")
        projections = load_projections(projections_csv)
        print(f"   → {len(projections)} uids with projection data\n")
    elif images_dir_path and not projections_csv:
        print("⚠  --images-dir provided but --projections-csv is missing. Reference images disabled.\n")

    # ── Load reports ─────────────────────────────────────────────────────
    print(f"📂 Loading reports from {csv_path} ...")
    records = load_reports(csv_path, limit=limit, offset=offset, projections=projections)
    print(f"   → {len(records)} reports loaded\n")

    # ── Shard: keep only the uids this process owns ───────────────────────
    shard_uids = _apply_shard([r.uid for r in records], num_shards, shard_index)
    if shard_uids is not None:
        before = len(records)
        records = [r for r in records if r.uid in shard_uids]
        print(
            f"🔀 Shard {shard_index}/{num_shards}: {len(records)} of {before} "
            f"report(s) assigned to this shard (uid % {num_shards} == {shard_index}).\n"
        )

    # ── Resume: skip uids already done in the target metadata file ────────
    results: list[dict] = []
    if resume:
        by_uid, completed_uids = _load_completed_uids(config.METADATA_FILE)
        if completed_uids:
            before = len(records)
            records = [r for r in records if r.uid not in completed_uids]
            print(
                f"↻  Resume: {len(completed_uids)} uid(s) already complete in "
                f"{config.METADATA_FILE.name}, skipping. "
                f"{len(records)} of {before} remaining.\n"
            )
            results = [by_uid[uid] for uid in by_uid if uid in completed_uids]
        elif Path(config.METADATA_FILE).exists():
            print(f"↻  Resume: {config.METADATA_FILE.name} exists but has no complete entries to skip.\n")

    # ── Build the LangChain chain once (reused across all records) ────────
    print(f"🤖 Initializing LLM chain (model: {config.CHAT_MODEL}) ...")
    chain = build_prompt_chain()
    print("   → Chain ready\n")

    # ── Preload SD3 before the loop ──────────────────────────────────────
    # SD3 is a multi-gigabyte local model, not an HTTP client. Loading it here
    # keeps the 30-90s cost off the first record and makes the tqdm ETA honest.
    # In img2img mode both cache entries are warmed so a mid-run downgrade to
    # txt2img never stalls on a cold load.
    if gen == "sd3" and not skip_image_generation:
        from .sd3_generator import load_sd3_pipeline

        load_sd3_pipeline("txt2img")
        if effective_sd3_mode == "img2img":
            load_sd3_pipeline("img2img")
        print()

    for record in tqdm(records, desc="Processing reports", unit="report"):
        entry: dict = {"uid": record.uid}

        # ── Step 1: Extract structured prompt via LLM ────────────────────
        try:
            structured = extract_structured_prompt(record, chain=chain)
            # Carry reference image metadata into the structured prompt
            structured.reference_images = record.reference_images
            entry["structured_prompt"] = structured.model_dump(
                exclude={"reference_images", "view"}
            )
        except Exception as e:
            print(f"\n  ✗ Failed to extract prompt for uid={record.uid}: {e}")
            entry["error_prompt"] = str(e)
            results.append(entry)
            _save_metadata(results)
            continue

        # ── Step 1.5: Split by views ─────────────────────────────────────
        view_prompts = split_prompt_by_views(structured, record.image)
        if len(view_prompts) > 1:
            view_names = [v for v, _ in view_prompts]
            tqdm.write(
                f"  📐 uid={record.uid} → {len(view_prompts)} views detected: {view_names}"
            )

        # ── Step 2: Resolve reference image paths (once per report) ──────
        ref_image_paths: list[Path] | None = None
        source_dimensions: tuple[int, int] | None = None
        matched_ratio: str | None = None

        if images_dir_path and record.reference_images:
            resolved = []
            for proj in record.reference_images:
                img_path = images_dir_path / proj.filename
                if img_path.exists():
                    resolved.append(img_path)
                else:
                    tqdm.write(f"  ⚠ Reference image not found: {img_path}")
            if resolved:
                ref_image_paths = resolved
                ref_views = ", ".join(
                    p.projection
                    for p in record.reference_images
                    if (images_dir_path / p.filename).exists()
                )
                tqdm.write(
                    f"  📎 uid={record.uid} → {len(resolved)} reference image(s) ({ref_views})"
                )

                # ── Read original image dimensions ────────────────────────
                try:
                    with PILImage.open(resolved[0]) as ref_img:
                        source_dimensions = ref_img.size  # (width, height)
                    matched_ratio = compute_best_aspect_ratio(*source_dimensions)
                    tqdm.write(
                        f"  📐 uid={record.uid} → source: {source_dimensions[0]}×{source_dimensions[1]}, "
                        f"matched aspect ratio: {matched_ratio}"
                    )
                except Exception as dim_err:
                    tqdm.write(
                        f"  ⚠ Could not read dimensions from {resolved[0].name}: {dim_err}"
                    )

            entry["reference_images"] = [
                {"filename": p.filename, "projection": p.projection}
                for p in record.reference_images
            ]
            if source_dimensions:
                entry["source_dimensions"] = {"width": source_dimensions[0], "height": source_dimensions[1]}
                entry["matched_aspect_ratio"] = matched_ratio

        # ── Propagate dimension info into structured prompt ───────────────
        if source_dimensions and matched_ratio:
            structured.source_dimensions = source_dimensions
            structured.matched_aspect_ratio = matched_ratio

        # ── Step 3: Generate images for each view ────────────────────────
        views_data: list[dict] = []
        for view_name, view_prompt in view_prompts:
            view_entry: dict = {"view": view_name}

            # Format the image generation prompt for this specific view
            image_prompt = format_image_prompt(view_prompt)
            view_entry["image_prompt"] = image_prompt

            if not skip_image_generation:
                try:
                    image_path = generate_image(
                        image_prompt,
                        record.uid,
                        generator=gen,
                        image_paths=ref_image_paths,
                        view_suffix=view_name,
                        source_dimensions=source_dimensions,
                        structured_prompt=view_prompt,
                        sd3_mode=sd3_mode,
                    )
                    view_entry["image_path"] = str(image_path)
                    if gen == "sd3":
                        view_entry["sd3_params"] = _sd3_params(
                            sd3_mode, ref_image_paths, source_dimensions
                        )
                    tqdm.write(
                        f"  ✓ uid={record.uid} [{view_name or 'single'}] → {image_path.name}"
                    )
                except Exception as e:
                    print(
                        f"\n  ✗ Image generation failed for uid={record.uid}"
                        f" [{view_name or 'single'}]: {e}"
                    )
                    view_entry["error_image"] = str(e)
            else:
                tqdm.write(
                    f"  ✓ uid={record.uid} [{view_name or 'single'}]"
                    " → prompt generated (image skipped)"
                )

            views_data.append(view_entry)

        entry["generator"] = gen
        entry["views"] = views_data
        results.append(entry)
        _save_metadata(results)

    print(f"\n📄 Metadata saved to {config.METADATA_FILE}")
    print(f"🖼  Images saved to {config.IMAGES_DIR}")

    return results


_DEFAULT_CSV = str(config.PROJECT_ROOT / "indiana_reports.csv")


def run_from_prompts(
    prompts_path: str | Path,
    generator: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
    images_dir: str | Path | None = None,
    sd3_mode: str | None = None,
    output_subdir: str | None = None,
    resume: bool = False,
    num_shards: int | None = None,
    shard_index: int | None = None,
) -> list[dict]:
    """
    Generate images from prompts already extracted into a metadata JSON file.

    Skips the CSV load and the LLM entirely, so this needs no OpenAI or Google
    credentials. The intended split is a cheap CPU run that extracts prompts
    (--skip-images) followed by a GPU run that only diffuses — on Kaggle that
    keeps LLM latency off the metered GPU session.

    Parameters
    ----------
    prompts_path  : a metadata.json written by a previous run
    images_dir    : needed only for SD3 img2img / Gemini reference images;
                    filenames come from the metadata's reference_images
    output_subdir : write elsewhere so the source file is not overwritten
    resume        : skip uids that already have a complete, error-free entry
                    in the target metadata file; see run_pipeline's resume.
    num_shards, shard_index : split entries by uid % num_shards == shard_index
                    for multi-GPU parallel runs; see run_pipeline.

    Returns
    -------
    list of metadata dicts, in the same shape as run_pipeline's
    """
    gen = generator or config.IMAGE_GENERATOR
    images_dir_path = Path(images_dir) if images_dir else None

    if output_subdir:
        _apply_output_subdir(output_subdir)
    elif Path(prompts_path).resolve() == config.METADATA_FILE.resolve():
        # Never clobber the file being read halfway through a long run.
        _apply_output_subdir("regenerated")
        print(
            "⚠  --prompts-from points at the default metadata file; writing to "
            f"{config.METADATA_FILE} instead to avoid overwriting it.\n"
        )

    print(f"📂 Loading prompts from {prompts_path} ...")
    with open(prompts_path, encoding="utf-8") as f:
        source_entries = json.load(f)

    entries = [e for e in source_entries if e.get("views")]

    # ── Shard: keep only the uids this process owns ───────────────────────
    shard_uids = _apply_shard([e["uid"] for e in entries], num_shards, shard_index)
    if shard_uids is not None:
        before = len(entries)
        entries = [e for e in entries if e["uid"] in shard_uids]
        print(
            f"🔀 Shard {shard_index}/{num_shards}: {len(entries)} of {before} "
            f"report(s) assigned to this shard (uid % {num_shards} == {shard_index}).\n"
        )

    if offset:
        entries = entries[offset:]
    if limit:
        entries = entries[:limit]
    print(f"   → {len(entries)} report(s) with prompts\n")

    # ── Resume: skip uids already done in the target metadata file ────────
    results: list[dict] = []
    if resume:
        by_uid, completed_uids = _load_completed_uids(config.METADATA_FILE)
        if completed_uids:
            before = len(entries)
            entries = [e for e in entries if e["uid"] not in completed_uids]
            print(
                f"↻  Resume: {len(completed_uids)} uid(s) already complete in "
                f"{config.METADATA_FILE.name}, skipping. "
                f"{len(entries)} of {before} remaining.\n"
            )
            results = [by_uid[uid] for uid in by_uid if uid in completed_uids]
        elif Path(config.METADATA_FILE).exists():
            print(f"↻  Resume: {config.METADATA_FILE.name} exists but has no complete entries to skip.\n")

    effective_sd3_mode = (sd3_mode or config.SD3_MODE).lower() if gen == "sd3" else None
    if effective_sd3_mode == "img2img" and not images_dir_path:
        print(
            "⚠  SD3_MODE=img2img but no --images-dir was given — every record "
            "will fall back to txt2img.\n"
        )

    if gen == "sd3":
        from .sd3_generator import load_sd3_pipeline

        load_sd3_pipeline("txt2img")
        if effective_sd3_mode == "img2img":
            load_sd3_pipeline("img2img")
        print()

    for source in tqdm(entries, desc="Generating images", unit="report"):
        uid = source["uid"]
        entry: dict = {"uid": uid, "generator": gen}

        # Reference images and geometry are replayed from the metadata rather
        # than re-derived, so a run matches the one that produced the prompts.
        ref_image_paths: list[Path] | None = None
        if images_dir_path and source.get("reference_images"):
            resolved = [
                images_dir_path / ref["filename"]
                for ref in source["reference_images"]
                if (images_dir_path / ref["filename"]).exists()
            ]
            if resolved:
                ref_image_paths = resolved
            entry["reference_images"] = source["reference_images"]

        source_dimensions: tuple[int, int] | None = None
        if source.get("source_dimensions"):
            dims = source["source_dimensions"]
            source_dimensions = (dims["width"], dims["height"])
            entry["source_dimensions"] = dims

        views_data: list[dict] = []
        for source_view in source["views"]:
            image_prompt = source_view.get("image_prompt")
            if not image_prompt:
                continue

            view_name = source_view.get("view")
            view_entry: dict = {"view": view_name, "image_prompt": image_prompt}

            try:
                image_path = generate_image(
                    image_prompt,
                    uid,
                    generator=gen,
                    image_paths=ref_image_paths,
                    view_suffix=view_name,
                    source_dimensions=source_dimensions,
                    sd3_mode=sd3_mode,
                )
                view_entry["image_path"] = str(image_path)
                if gen == "sd3":
                    view_entry["sd3_params"] = _sd3_params(
                        sd3_mode, ref_image_paths, source_dimensions
                    )
                tqdm.write(
                    f"  ✓ uid={uid} [{view_name or 'single'}] → {image_path.name}"
                )
            except Exception as e:
                print(
                    f"\n  ✗ Image generation failed for uid={uid}"
                    f" [{view_name or 'single'}]: {e}"
                )
                view_entry["error_image"] = str(e)

            views_data.append(view_entry)

        entry["views"] = views_data
        results.append(entry)
        _save_metadata(results)

    print(f"\n📄 Metadata saved to {config.METADATA_FILE}")
    print(f"🖼  Images saved to {config.IMAGES_DIR}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Synthetic Radiology Image Generation Pipeline"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=_DEFAULT_CSV,
        help="Path to the Indiana reports CSV file",
    )
    parser.add_argument(
        "--projections-csv",
        type=str,
        default=None,
        help=(
            "Path to indiana_projections.csv. "
            "Required when --images-dir is provided to enable reference image input."
        ),
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help=(
            "Directory containing original X-ray PNG images. "
            "On Kaggle: /kaggle/input/chest-xrays-indiana-university/images/images_normalized. "
            "When provided with --projections-csv, images are passed as multimodal input to Gemini."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of reports to process (default: all)",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=None,
        help="Skip the first N reports in the dataset",
    )
    parser.add_argument(
        "--generator",
        type=str,
        choices=["dalle", "gemini", "sd3"],
        default=None,
        help="Image generator backend (overrides .env config)",
    )
    parser.add_argument(
        "--sd3-mode",
        type=str,
        choices=["txt2img", "img2img"],
        default=None,
        help=(
            "SD3 generation mode. txt2img ignores reference images and produces "
            "purely synthetic output; img2img conditions on the real X-ray "
            "(requires --images-dir). Overrides SD3_MODE in .env."
        ),
    )
    parser.add_argument(
        "--sd3-strength",
        type=float,
        default=None,
        help=(
            "SD3 img2img denoising strength, 0.0-1.0. Lower keeps more of the "
            "reference X-ray; below ~0.4 the output is barely synthetic. "
            "Default 0.75. Ignored in txt2img mode."
        ),
    )
    parser.add_argument(
        "--sd3-steps",
        type=int,
        default=None,
        help="SD3 inference steps (default 28)",
    )
    parser.add_argument(
        "--sd3-guidance",
        type=float,
        default=None,
        help="SD3 guidance scale / CFG (default 7.0)",
    )
    parser.add_argument(
        "--sd3-seed",
        type=int,
        default=None,
        help="SD3 base seed; each uid is offset from it so runs are reproducible",
    )
    parser.add_argument(
        "--sd3-model-id",
        type=str,
        default=None,
        help="HuggingFace model id for SD3 (default stabilityai/stable-diffusion-3-medium-diffusers)",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=None,
        help=(
            "Write images to output/images/<NAME>/ and metadata to "
            "output/metadata_<NAME>.json, so runs do not overwrite each other."
        ),
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Only generate prompts, skip image generation (for testing)",
    )
    parser.add_argument(
        "--prompts-from",
        type=str,
        default=None,
        help=(
            "Generate images from an existing metadata.json instead of reading "
            "the CSV and calling the LLM. Needs no LLM credentials, so prompt "
            "extraction can run on CPU and only generation on the GPU. "
            "Mutually exclusive with --csv."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip uids that already have a complete, error-free entry in the "
            "target metadata file (see --output-subdir), and continue "
            "appending into it. Failed or partial entries are retried. "
            "Metadata is saved after every record regardless of this flag, so "
            "an interrupted run only needs the same command re-run with "
            "--resume to pick back up."
        ),
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help=(
            "Split the work into N shards for parallel multi-GPU runs (e.g. "
            "Kaggle's GPU T4 x2). Requires --shard-index. Run one process per "
            "GPU with CUDA_VISIBLE_DEVICES set and a distinct --output-subdir "
            "per shard, then combine with merge_shards.py."
        ),
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help=(
            "Which shard this process handles, 0-indexed. Requires "
            "--num-shards. Assignment is uid %% num_shards == shard_index, "
            "so it is stable regardless of CSV ordering."
        ),
    )

    args = parser.parse_args()

    if (args.num_shards is None) != (args.shard_index is None):
        parser.error("--num-shards and --shard-index must be given together.")
    if args.num_shards and not args.output_subdir:
        print(
            "⚠  Sharding without --output-subdir: every shard will write to the "
            "same metadata.json and image directory. Give each shard its own "
            "--output-subdir, then merge with merge_shards.py.\n"
        )

    # CLI overrides config, following the value = arg or config.DEFAULT pattern.
    if args.sd3_steps is not None:
        config.SD3_STEPS = args.sd3_steps
    if args.sd3_guidance is not None:
        config.SD3_GUIDANCE_SCALE = args.sd3_guidance
    if args.sd3_seed is not None:
        config.SD3_SEED = args.sd3_seed
    if args.sd3_model_id is not None:
        config.SD3_MODEL_ID = args.sd3_model_id
    if args.sd3_strength is not None:
        effective_mode = (args.sd3_mode or config.SD3_MODE).lower()
        if effective_mode == "txt2img":
            print("⚠  --sd3-strength has no effect in txt2img mode; ignoring.\n")
        config.SD3_IMG2IMG_STRENGTH = args.sd3_strength

    if args.prompts_from:
        if args.csv != _DEFAULT_CSV:
            print("⚠  --csv is ignored when --prompts-from is given.\n")
        if args.skip_images:
            parser.error("--skip-images makes --prompts-from a no-op.")
        run_from_prompts(
            prompts_path=args.prompts_from,
            generator=args.generator,
            limit=args.limit,
            offset=args.offset,
            images_dir=args.images_dir,
            sd3_mode=args.sd3_mode,
            output_subdir=args.output_subdir,
            resume=args.resume,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
        return

    run_pipeline(
        csv_path=args.csv,
        limit=args.limit,
        offset=args.offset,
        generator=args.generator,
        skip_image_generation=args.skip_images,
        projections_csv=args.projections_csv,
        images_dir=args.images_dir,
        sd3_mode=args.sd3_mode,
        output_subdir=args.output_subdir,
        resume=args.resume,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
    )


if __name__ == "__main__":
    main()

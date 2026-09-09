"""
Merge sharded pipeline output back into one metadata.json / images/ dir.

Running two shards (e.g. one per GPU on a Kaggle GPU T4 x2 session) via
`--num-shards 2 --shard-index 0/1 --output-subdir gpu0/gpu1` leaves results in
output/metadata_gpu0.json + output/images/gpu0/ and output/metadata_gpu1.json
+ output/images/gpu1/. This combines them into a single output/metadata.json
and output/images/ directory, sorted by uid, so downstream steps (dataset
export, --prompts-from, the img2img comparison cells) see one normal run.

Usage
-----
    python merge_shards.py gpu0 gpu1
    python merge_shards.py gpu0 gpu1 --output-subdir merged   # merge elsewhere instead
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from pipeline import config


def merge_shards(shard_names: list[str], output_subdir: str | None = None) -> Path:
    # Built from config.OUTPUT_DIR rather than config.IMAGES_DIR/METADATA_FILE:
    # those two are mutated by _apply_output_subdir, so reusing them here would
    # pick up whichever shard ran last instead of the intended merge target.
    if output_subdir:
        dest_images = config.OUTPUT_DIR / "images" / output_subdir
        dest_metadata = config.OUTPUT_DIR / f"metadata_{output_subdir}.json"
    else:
        dest_images = config.OUTPUT_DIR / "images"
        dest_metadata = config.OUTPUT_DIR / "metadata.json"
    dest_images.mkdir(parents=True, exist_ok=True)

    merged: dict[int, dict] = {}
    seen_files = 0

    for name in shard_names:
        shard_metadata = config.OUTPUT_DIR / f"metadata_{name}.json"
        shard_images = config.OUTPUT_DIR / "images" / name

        if not shard_metadata.exists():
            print(f"⚠  {shard_metadata} not found — skipping shard '{name}'")
            continue

        with open(shard_metadata, encoding="utf-8") as f:
            entries = json.load(f)

        for entry in entries:
            uid = entry["uid"]
            if uid in merged:
                print(f"⚠  uid={uid} present in more than one shard — keeping the last one seen "
                      f"(shard '{name}'). Shards were supposed to be disjoint; check --num-shards "
                      "was identical across all shard runs.")
            merged[uid] = entry
        print(f"📄 {name}: {len(entries)} entries from {shard_metadata.name}")

        if shard_images.exists():
            for img_path in shard_images.glob("*.png"):
                dest_path = dest_images / img_path.name
                if img_path.resolve() != dest_path.resolve():
                    shutil.copy2(img_path, dest_path)
                seen_files += 1
        else:
            print(f"⚠  {shard_images} not found — no images copied for shard '{name}'")

    results = [merged[uid] for uid in sorted(merged)]
    with open(dest_metadata, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(
        f"\n✅ Merged {len(shard_names)} shard(s): {len(results)} report(s), "
        f"{seen_files} image file(s) copied."
    )
    print(f"   → {dest_metadata}")
    print(f"   → {dest_images}")
    return dest_metadata


def main():
    parser = argparse.ArgumentParser(description="Merge sharded pipeline output")
    parser.add_argument(
        "shards",
        nargs="+",
        help="The --output-subdir name(s) used for each shard, e.g.: gpu0 gpu1",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=None,
        help="Merge into output/metadata_<NAME>.json instead of the default output/metadata.json",
    )
    args = parser.parse_args()

    if len(args.shards) < 2:
        print("⚠  Only one shard name given — merging is a no-op copy/rename.")

    merge_shards(args.shards, output_subdir=args.output_subdir)


if __name__ == "__main__":
    sys.exit(main())

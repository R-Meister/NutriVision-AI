#!/usr/bin/env python3
"""
Download the real 12-class food dataset used to train the classifiers.

This uses the Hugging Face dataset viewer's server-side filter endpoint, so it
only transfers the images that are actually needed (~300 MB) instead of
downloading several GB of parquet shards.

Sources and class mapping
-------------------------
- Food-101            → pizza, burger, french fries, samosa, fried rice,
                        club sandwich, omelette
- Indian Foods        → biryani, dosa
- Indian Food Images  → idli
- Fruits-360          → apple, banana

Output layout::

    data/food_images/{train,val,test}/<class>/<class>_<source>_<idx>.jpg
    data/food_images/metadata.json
    assets/samples/<class>.jpg          (one test image per class)

Usage:
    python training/download_dataset.py
    python training/download_dataset.py --per-class 150 --force
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from PIL import Image, ImageOps

OUT_DIR = config.REAL_DATA_DIR
SAMPLES_DIR = os.path.join(config.ASSETS_DIR, "samples")
API = "https://datasets-server.huggingface.co/filter"

TRAIN_PER_CLASS = 200
VAL_PER_CLASS = 40
TEST_PER_CLASS = 47

# Human-readable source metadata stored in the manifest.
SOURCES = {
    "food101": {
        "name": "Food-101",
        "homepage": "https://huggingface.co/datasets/ethz/food101",
        "license": "Research use (see dataset card)",
    },
    "indian_foods": {
        "name": "Indian Foods Dataset",
        "homepage": "https://huggingface.co/datasets/bharat-raghunathan/indian-foods-dataset",
        "license": "See dataset card",
    },
    "indian_food_images": {
        "name": "Indian Food Images",
        "homepage": "https://huggingface.co/datasets/rajistics/indian_food_images",
        "license": "See dataset card",
    },
    "fruits360": {
        "name": "Fruits-360",
        "homepage": "https://huggingface.co/datasets/PedroSampaio/fruits-360",
        "license": "CC BY-SA 4.0 (see dataset card)",
    },
}

# class → (dataset repo, label index, source key, split name for test)
CLASS_SPECS = {
    "pizza":          ("ethz/food101", 76, "food101", "validation"),
    "burger":         ("ethz/food101", 53, "food101", "validation"),
    "french_fries":   ("ethz/food101", 40, "food101", "validation"),
    "samosa":         ("ethz/food101", 85, "food101", "validation"),
    "fried_rice":     ("ethz/food101", 44, "food101", "validation"),
    "sandwich":       ("ethz/food101", 25, "food101", "validation"),
    "omelette":       ("ethz/food101", 67, "food101", "validation"),
    "biryani":        ("bharat-raghunathan/indian-foods-dataset", 0, "indian_foods", "test"),
    "dosa":           ("rajistics/indian_food_images", 13, "indian_food_images", "test"),
    "idli":           ("rajistics/indian_food_images", 8, "indian_food_images", "test"),
    "apple":          ("PedroSampaio/fruits-360", 5, "fruits360", "test"),
    "banana":         ("PedroSampaio/fruits-360", 11, "fruits360", "test"),
}

# Extra cultivars merged into one app class.  Food-101/Indian datasets use a
# single label; Fruit classes are combined in `collect_fruit`.
FRUIT_EXTRA = {
    "apple":  [("PedroSampaio/fruits-360", 0), ("PedroSampaio/fruits-360", 6)],
    "banana": [("PedroSampaio/fruits-360", 12)],
}

# Fruits-360 is small enough (~380 MB) to download as parquet; the dataset
# viewer's filter index for its 67k images is slow to warm up.
FRUIT_PARQUET = {
    "train": "train-00000-of-00001-086dc013443c484c.parquet",
    "test": "test-00000-of-00001-0d294abe3826b2e6.parquet",
}
FRUIT_PARQUET_BASE = (
    "https://huggingface.co/datasets/PedroSampaio/fruits-360/resolve/main/data/"
)

_LAST_REQUEST = 0.0
_MIN_INTERVAL = 2.0  # seconds between API calls (dataset viewer rate limit)


def _request_json(url: str, retries: int = 10) -> dict:
    global _LAST_REQUEST
    last_error = None
    for attempt in range(1, retries + 1):
        wait = _MIN_INTERVAL - (time.monotonic() - _LAST_REQUEST)
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST = time.monotonic()
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "NutriVisionAI/1.0"}
            )
            with urllib.request.urlopen(req, timeout=90) as response:
                body = response.read().decode("utf-8")
            return json.loads(body)
        except (urllib.error.HTTPError, urllib.error.URLError,
                TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            # 500/429/503 and viewer cold starts are retried with backoff.
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"Dataset API failed after {retries} attempts: {last_error}")


def _filter_rows(repo: str, split: str, label: int, offset: int,
                 length: int = 100) -> dict:
    params = {
        "dataset": repo,
        "config": "default",
        "split": split,
        "where": f'"label"={label}',
        "offset": str(offset),
        "length": str(length),
    }
    return _request_json(API + "?" + urllib.parse.urlencode(params))


def _image_urls(repo: str, split: str, label: int, target: int) -> list[str]:
    """Return up to ``target`` image URLs for one label."""
    urls: list[str] = []
    offset = 0
    while len(urls) < target:
        data = _filter_rows(repo, split, label, offset, 100)
        rows = data.get("rows", [])
        if not rows:
            break
        urls.extend(row["row"]["image"]["src"] for row in rows)
        offset += len(rows)
        total = data.get("num_rows_total", 0)
        if offset >= total:
            break
    return urls[:target]


def _download_image(url: str, path: str, retries: int = 4) -> None:
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "NutriVisionAI/1.0"}
            )
            with urllib.request.urlopen(req, timeout=60) as response:
                raw = response.read()
            img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
            if max(img.size) > 640:
                img.thumbnail((640, 640), Image.LANCZOS)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            img.save(path, "JPEG", quality=92)
            return
        except Exception as exc:  # network / expired signed URL
            last_error = exc
            time.sleep(2 * attempt)
    raise RuntimeError(f"Could not download {path}: {last_error}")


def _save_many(items: list[tuple[str, str]]) -> None:
    """Download (url, path) pairs concurrently."""
    failures = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_download_image, u, p): p for u, p in items}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                failures += 1
                print(f"  ⚠️  {exc}")
    if failures:
        print(f"  ⚠️  {failures} image(s) failed; continuing with the rest.")


def download_class(food: str, repo: str, label: int, source: str,
                   test_split: str, per_class: int) -> tuple[int, int, int]:
    """Download train/val/test images for one class. Returns counts."""
    need_train = per_class
    need_val = VAL_PER_CLASS
    need_test = TEST_PER_CLASS
    pool = need_train + need_val

    print(f"  requesting train URLs ({pool}) …", flush=True)
    urls = _image_urls(repo, "train", label, pool)
    if len(urls) < pool:
        print(f"  ⚠️  {food}: only {len(urls)} train rows available "
              f"(wanted {pool}); reducing train/val proportionally.", flush=True)
        if len(urls) < 2:
            raise RuntimeError(f"Not enough training rows for {food}")
        n_val = max(1, min(need_val, len(urls) // 6))
        need_train = len(urls) - n_val
        need_val = n_val
    cut = min(need_train, len(urls))
    train_urls = urls[:cut]
    val_urls = urls[cut:cut + need_val]

    print(f"  requesting test URLs ({need_test}) …", flush=True)
    test_urls = _image_urls(repo, test_split, label, need_test)

    print(f"  downloading {len(train_urls) + len(val_urls) + len(test_urls)} "
          f"images …", flush=True)

    tasks = []
    for i, url in enumerate(train_urls):
        tasks.append((url, os.path.join(OUT_DIR, "train", food,
                                        f"{food}_{source}_{i:05d}.jpg")))
    for i, url in enumerate(val_urls):
        tasks.append((url, os.path.join(OUT_DIR, "val", food,
                                        f"{food}_{source}_{i:05d}.jpg")))
    for i, url in enumerate(test_urls):
        tasks.append((url, os.path.join(OUT_DIR, "test", food,
                                        f"{food}_{source}_{i:05d}.jpg")))
    _save_many(tasks)
    return len(train_urls), len(val_urls), len(test_urls)


FRUIT_LABELS = {
    "apple": {0, 5, 6},      # Braeburn, Red, Red Delicious
    "banana": {11, 12},      # Banana, Banana Lady Finger
}


def _decode_bytes(data: bytes):
    try:
        return ImageOps.exif_transpose(
            Image.open(io.BytesIO(data))
        ).convert("RGB")
    except Exception:
        return None


def _save_image(img: Image.Image, split: str, food: str, source: str,
                idx: int) -> None:
    folder = os.path.join(OUT_DIR, split, food)
    os.makedirs(folder, exist_ok=True)
    if max(img.size) > 640:
        img.thumbnail((640, 640), Image.LANCZOS)
    img.save(os.path.join(folder, f"{food}_{source}_{idx:05d}.jpg"),
             "JPEG", quality=92)


def _ensure_fruit_parquet() -> dict[str, str]:
    """Download the two Fruits-360 parquet files (~380 MB) if needed."""
    folder = os.path.join(config.DATA_DIR, "raw", "fruits360")
    os.makedirs(folder, exist_ok=True)
    paths = {}
    for split, filename in FRUIT_PARQUET.items():
        path = os.path.join(folder, filename)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print(f"  downloading Fruits-360 {split} parquet …", flush=True)
            urllib.request.urlretrieve(FRUIT_PARQUET_BASE + filename, path)
        paths[split] = path
    return paths


def _collect_fruit_split(path: str, limits: dict[str, int]):
    import pyarrow.parquet as pq

    collected = defaultdict(list)
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=64, columns=["image", "label"]):
        images = batch.column("image")
        labels = batch.column("label")
        for i in range(batch.num_rows):
            label = int(labels[i].as_py())
            food = next((f for f, ls in FRUIT_LABELS.items() if label in ls), None)
            if food is None or len(collected[food]) >= limits[food]:
                continue
            raw = images[i].as_py()
            if isinstance(raw, dict):
                raw = raw.get("bytes")
            img = _decode_bytes(raw) if raw else None
            if img is not None:
                collected[food].append(img)
        if all(len(collected[f]) >= limits[f] for f in FRUIT_LABELS):
            break
    return collected


def download_fruits_from_parquet(per_class: int) -> None:
    """Merge several Fruits-360 cultivars into the apple/banana classes."""
    paths = _ensure_fruit_parquet()
    train_pool = _collect_fruit_split(
        paths["train"], {f: per_class + VAL_PER_CLASS for f in FRUIT_LABELS}
    )
    test_pool = _collect_fruit_split(
        paths["test"], {f: TEST_PER_CLASS for f in FRUIT_LABELS}
    )

    for food in FRUIT_LABELS:
        imgs = train_pool.get(food, [])
        if len(imgs) < per_class + VAL_PER_CLASS:
            n_val = max(1, min(VAL_PER_CLASS, len(imgs) // 6))
            n_train = max(0, len(imgs) - n_val)
        else:
            n_train, n_val = per_class, VAL_PER_CLASS

        for i, img in enumerate(imgs[:n_train]):
            _save_image(img, "train", food, "fruits360", i)
        for i, img in enumerate(imgs[n_train:n_train + n_val]):
            _save_image(img, "val", food, "fruits360", i)
        for i, img in enumerate(test_pool.get(food, [])[:TEST_PER_CLASS]):
            _save_image(img, "test", food, "fruits360", i)
        print(f"  {food}: train={n_train} val={n_val} "
              f"test={len(test_pool.get(food, [])[:TEST_PER_CLASS])}")


def copy_samples() -> None:
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    for food in config.FOOD_CLASSES:
        folder = os.path.join(OUT_DIR, "test", food)
        if not os.path.isdir(folder):
            continue
        files = sorted(f for f in os.listdir(folder) if f.endswith(".jpg"))
        if files:
            shutil.copyfile(
                os.path.join(folder, files[0]),
                os.path.join(SAMPLES_DIR, f"{food}.jpg"),
            )


def _class_complete(food: str) -> bool:
    for split in ("train", "val", "test"):
        folder = os.path.join(OUT_DIR, split, food)
        if not os.path.isdir(folder):
            return False
        if not any(f.lower().endswith((".jpg", ".jpeg", ".png"))
                   for f in os.listdir(folder)):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=TRAIN_PER_CLASS)
    parser.add_argument("--force", action="store_true",
                        help="Delete data/food_images before downloading")
    args = parser.parse_args()
    per_class = args.per_class

    if os.path.isdir(OUT_DIR):
        if args.force:
            shutil.rmtree(OUT_DIR)
        else:
            print(f"Resuming in {OUT_DIR} (existing classes are skipped).")
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Downloading {len(CLASS_SPECS)} classes → {OUT_DIR}")
    print(f"Split sizes: train={per_class} val={VAL_PER_CLASS} "
          f"test={TEST_PER_CLASS} per class\n")

    counts = defaultdict(dict)
    fruits_done = False
    for n, food in enumerate(config.FOOD_CLASSES, 1):
        if not args.force and _class_complete(food):
            print(f"[{n:2d}/{len(config.FOOD_CLASSES)}] {food}  (already complete)")
            continue

        if food in FRUIT_EXTRA:
            print(f"[{n:2d}/{len(config.FOOD_CLASSES)}] {food}  (fruits360)")
            if not fruits_done:
                download_fruits_from_parquet(per_class)
                fruits_done = True
            continue

        repo, label, source, test_split = CLASS_SPECS[food]
        print(f"[{n:2d}/{len(config.FOOD_CLASSES)}] {food}  ({source})")
        tr, va, te = download_class(
            food, repo, label, source, test_split, per_class
        )
        print(f"  {food}: train={tr} val={va} test={te}")

    counts = defaultdict(dict)
    for food in config.FOOD_CLASSES:
        for split in ("train", "val", "test"):
            folder = os.path.join(OUT_DIR, split, food)
            counts[split][food] = (
                len(os.listdir(folder)) if os.path.isdir(folder) else 0
            )

    copy_samples()

    manifest = {
        "classes": config.FOOD_CLASSES,
        "splits": counts,
        "sources": SOURCES,
        "seed": config.RANDOM_SEED,
        "note": (
            "Real photographs from public datasets. Food-101 validation is used "
            "as the held-out test set; other datasets use their official test split."
        ),
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n✅ Dataset ready:", OUT_DIR)
    for split in ("train", "val", "test"):
        print(f"  {split}: {sum(counts[split].values())} images")
    return 0


if __name__ == "__main__":
    sys.exit(main())

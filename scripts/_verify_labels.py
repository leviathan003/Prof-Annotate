"""
scripts/_verify_labels.py
Standalone verification for the label-materialization fix. Not part of the app;
run with the project venv:  python scripts/_verify_labels.py

Covers materialize_empty_labels (none/partial/all labelled, idempotency,
no-clobber, empty-split dir creation) and label_path_for_image (structure
mapping + the derive_label_path .index("images") mis-map regression).
"""

from __future__ import annotations

import os
import stat
import sys
import tempfile
import time
from pathlib import Path

import yaml

from profannotate.core.annotation.writer import (
    label_path_for_image,
    materialize_empty_labels,
)
from profannotate.core.dataset.yaml_handler import load_yaml, save_yaml

PASS = 0
FAIL = 0


def check(cond: bool, msg: str) -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok   {msg}")
    else:
        FAIL += 1
        print(f"  FAIL {msg}")


def make_dataset(root: Path, train: int, val: int) -> None:
    """Create images/train + images/val with tiny placeholder image files."""
    for split, n in (("train", train), ("val", val)):
        d = root / "images" / split
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            (d / f"img_{i:03d}.jpg").write_bytes(b"\xff\xd8\xff\xd9")  # minimal JPEG-ish


def txts(root: Path, split: str) -> list[Path]:
    d = root / "labels" / split
    return sorted(d.glob("*.txt")) if d.is_dir() else []


def test_no_labels() -> None:
    print("test: dataset with NO label files")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_dataset(root, train=3, val=2)
        created = materialize_empty_labels(root)
        check(created == 5, f"created == 5 (got {created})")
        check((root / "labels" / "train").is_dir(), "labels/train created")
        check((root / "labels" / "val").is_dir(), "labels/val created")
        check(len(txts(root, "train")) == 3, "3 train .txt")
        check(len(txts(root, "val")) == 2, "2 val .txt")
        check(all(p.stat().st_size == 0 for p in txts(root, "train") + txts(root, "val")),
              "all .txt are 0 bytes")
        # stems match image stems
        img_stems = {p.stem for p in (root / "images" / "train").glob("*.jpg")}
        lbl_stems = {p.stem for p in txts(root, "train")}
        check(img_stems == lbl_stems, "train label stems match image stems")


def test_partial_labels() -> None:
    print("test: dataset with SOME labels (partial) — fill gaps, keep existing")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_dataset(root, train=4, val=0)
        # Pre-create one real (non-empty) label.
        lbl_dir = root / "labels" / "train"
        lbl_dir.mkdir(parents=True, exist_ok=True)
        existing = lbl_dir / "img_000.txt"
        existing.write_text("0 0.5 0.5 0.2 0.2\n")
        before = existing.read_text()

        created = materialize_empty_labels(root)
        check(created == 3, f"created == 3 (got {created}) — only the 3 missing")
        check(len(txts(root, "train")) == 4, "4 train .txt total")
        check(existing.read_text() == before, "existing non-empty label NOT clobbered")
        empties = [p for p in txts(root, "train") if p.name != "img_000.txt"]
        check(all(p.stat().st_size == 0 for p in empties), "the 3 new ones are empty")


def test_all_labels() -> None:
    print("test: dataset where ALL images already have labels — no-op")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_dataset(root, train=2, val=2)
        materialize_empty_labels(root)  # create them
        # Put content in all to ensure none get clobbered on the second pass.
        for split in ("train", "val"):
            for p in txts(root, split):
                p.write_text("0 0.1 0.1 0.1 0.1\n")
        created = materialize_empty_labels(root)
        check(created == 0, f"created == 0 on fully-labelled dataset (got {created})")
        check(all(p.stat().st_size > 0 for p in txts(root, "train") + txts(root, "val")),
              "no existing label was clobbered")


def test_idempotency() -> None:
    print("test: idempotency — second call creates nothing")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_dataset(root, train=5, val=5)
        first = materialize_empty_labels(root)
        second = materialize_empty_labels(root)
        check(first == 10, f"first run created 10 (got {first})")
        check(second == 0, f"second run created 0 (got {second})")


def test_empty_split_dir() -> None:
    print("test: split with images dir but zero images still gets labels dir")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "images" / "train").mkdir(parents=True)  # empty
        make_dataset(root, train=0, val=1)  # adds val image; train stays empty
        created = materialize_empty_labels(root)
        check((root / "labels" / "train").is_dir(), "labels/train created for empty split")
        check(created == 1, f"only the 1 val image got a label (got {created})")


def test_label_path_mapping() -> None:
    print("test: label_path_for_image structure mapping + mis-map regression")
    root = Path("/data/proj")
    got = label_path_for_image(root, root / "images" / "train" / "x.jpg")
    check(got == root / "labels" / "train" / "x.txt",
          f"images/train/x.jpg -> labels/train/x.txt (got {got})")

    # Regression: a root whose OWN path contains 'images' must not be mis-mapped
    # the way derive_label_path's .index('images') would.
    root2 = Path("/srv/images/dataset")
    got2 = label_path_for_image(root2, root2 / "images" / "val" / "y.png")
    check(got2 == root2 / "labels" / "val" / "y.txt",
          f"ancestor named 'images' not mis-mapped (got {got2})")

    # Fallback: image not under root -> derive_label_path behaviour.
    got3 = label_path_for_image(Path("/a/b"), Path("/c/images/train/z.jpg"))
    check(got3 == Path("/c/labels/train/z.txt"),
          f"out-of-root falls back to derive_label_path (got {got3})")


def test_speed() -> None:
    print("test: speed on a 2000-image dataset")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_dataset(root, train=1500, val=500)
        t0 = time.perf_counter()
        created = materialize_empty_labels(root)
        dt = time.perf_counter() - t0
        check(created == 2000, f"created 2000 (got {created})")
        check(dt < 5.0, f"materialized 2000 labels in {dt:.3f}s (< 5s)")
        # Idempotent second pass should be quick too.
        t1 = time.perf_counter()
        materialize_empty_labels(root)
        dt2 = time.perf_counter() - t1
        check(dt2 < 5.0, f"idempotent re-scan of 2000 in {dt2:.3f}s (< 5s)")


def test_readonly_dataset() -> None:
    print("test: read-only dataset — materialize degrades gracefully, no raise")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        d = root / "images" / "train"
        d.mkdir(parents=True)
        for i in range(3):
            (d / f"i{i}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        os.chmod(root, stat.S_IRUSR | stat.S_IXUSR)  # r-x: cannot create under root
        try:
            n = materialize_empty_labels(root)  # must not raise
            check(n == 0, f"read-only materialize returns 0 without raising (got {n})")
        except Exception as exc:  # noqa: BLE001
            check(False, f"read-only materialize raised {exc!r}")
        finally:
            os.chmod(root, stat.S_IRWXU)


def test_save_yaml_empty_body() -> None:
    print("test: save_yaml with empty body + kpt extras is valid YAML (regression)")
    with tempfile.TemporaryDirectory() as tmp:
        save_yaml(tmp, {"kpt_shape": [17, 3], "keypoint_names": ["a", "b"]})
        txt = (Path(tmp) / "data.yaml").read_text()
        try:
            parsed = yaml.safe_load(txt)
            check(parsed.get("keypoint_names") == ["a", "b"], "empty-body yaml parses + roundtrips")
        except Exception as exc:  # noqa: BLE001
            check(False, f"empty-body yaml failed to parse: {exc}")
        check(load_yaml(tmp).get("kpt_shape") == [17, 3], "load_yaml recovers kpt config")
    with tempfile.TemporaryDirectory() as tmp:
        save_yaml(tmp, {"path": tmp, "train": "images/train", "val": "images/val",
                        "nc": 2, "names": ["a", "b"], "kpt_shape": [3, 3],
                        "keypoint_names": ["x", "y", "z"]})
        d = yaml.safe_load((Path(tmp) / "data.yaml").read_text())
        check(d["nc"] == 2 and d["names"] == ["a", "b"] and d["keypoint_names"] == ["x", "y", "z"],
              "full-body yaml unchanged + valid")


def main() -> int:
    test_no_labels()
    test_partial_labels()
    test_all_labels()
    test_idempotency()
    test_empty_split_dir()
    test_label_path_mapping()
    test_readonly_dataset()
    test_save_yaml_empty_body()
    test_speed()
    print(f"\n{PASS} passed, {FAIL} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

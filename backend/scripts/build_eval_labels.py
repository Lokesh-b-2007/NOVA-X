"""Build OOD and open-vocabulary metric labels from organized test photos.

Run with the backend online, from backend/:
    python scripts/build_eval_labels.py
"""
import argparse
import csv
import json
import mimetypes
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings  # noqa: E402

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def post_image(api: str, path: Path, timeout: int) -> dict:
    boundary = uuid.uuid4().hex
    filename = path.name.replace('"', "")
    mime_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode(),
        f"Content-Type: {mime_type}\r\n\r\n".encode(),
        path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    request = urllib.request.Request(
        api.rstrip("/") + "/api/detect/image",
        data=body,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main():
    default_images = settings.KNOWN_DATASET_DIR / "eval_images"
    default_output = settings.KNOWN_DATASET_DIR / "eval"
    parser = argparse.ArgumentParser(description="Create labeled OOD/OVD evaluation CSV files.")
    parser.add_argument("--api", default="http://localhost:8001", help="Running backend base URL.")
    parser.add_argument("--images-dir", type=Path, default=default_images)
    parser.add_argument("--out-dir", type=Path, default=default_output)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()

    jobs = [(path, "known", None) for path in list_images(args.images_dir / "known")]
    unknown_root = args.images_dir / "unknown"
    jobs.extend((path, "unknown", None) for path in list_images(unknown_root))
    if unknown_root.is_dir():
        for label_dir in sorted(path for path in unknown_root.iterdir() if path.is_dir()):
            true_label = label_dir.name.strip().lower().replace("_", " ")
            if true_label not in {label.lower() for label in settings.OVD_VOCABULARY}:
                print(f"WARNING: '{true_label}' is not in NOVA_OVD_VOCABULARY; OVD accuracy cannot match it.")
            jobs.extend((path, "unknown", true_label) for path in list_images(label_dir))

    if not jobs:
        raise SystemExit(
            f"No test images found under {args.images_dir}. Add known photos to known/ "
            "and unknown photos to unknown/<true-label>/."
        )

    ood_rows, ovd_rows = [], []
    for index, (path, group, true_label) in enumerate(jobs, 1):
        relative = path.relative_to(args.images_dir)
        try:
            result = post_image(args.api, path, args.timeout)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")[:300]
            raise SystemExit(f"Backend returned HTTP {e.code} for {relative}: {detail}") from e
        except urllib.error.URLError as e:
            raise SystemExit(f"Could not reach backend at {args.api}: {e.reason}") from e

        detections = result.get("detections", [])
        is_unknown = int(group == "unknown")
        for detection in detections:
            ood_rows.append({
                "detection_id": detection["id"],
                "is_unknown": is_unknown,
                "image": str(relative),
            })
            if is_unknown and true_label and detection.get("novel_object_id"):
                ovd_rows.append({
                    "novel_object_id": detection["novel_object_id"],
                    "true_label": true_label,
                    "image": str(relative),
                })
        print(f"[{index}/{len(jobs)}] {relative}: {len(detections)} detection(s)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows, columns in (
        ("ood_labels.csv", ood_rows, ["detection_id", "is_unknown", "image"]),
        ("ovd_labels.csv", ovd_rows, ["novel_object_id", "true_label", "image"]),
    ):
        with (args.out_dir / filename).open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    known_count = sum(row["is_unknown"] == 0 for row in ood_rows)
    unknown_count = sum(row["is_unknown"] == 1 for row in ood_rows)
    print(f"Wrote {known_count} known and {unknown_count} unknown labels to {args.out_dir / 'ood_labels.csv'}")
    print(f"Wrote {len(ovd_rows)} open-vocabulary labels to {args.out_dir / 'ovd_labels.csv'}")
    if not known_count or not unknown_count:
        print("WARNING: OOD metrics require both known and unknown detections.")
    print("Refresh the Evaluation page to load the computed metrics.")


if __name__ == "__main__":
    main()

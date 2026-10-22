# Model evaluation data

The Evaluation page reports measured metrics only. The supplied `data.yaml`
uses Ultralytics COCO128 for a quick known-detector sanity check; COCO128 is
not held out, and the default YOLO weights were trained on COCO, so do not
report these numbers as an unbiased benchmark.

## Known detection

From `backend/`, with the backend virtual environment active, run:

```powershell
python scripts/run_known_eval.py
```

Ultralytics downloads COCO128 on first use if needed, validates configured
known classes, and caches results in `known_metrics.json`. The Evaluation page
reads the cache instead of running a long validation on every web request.
Re-run when weights, known classes, or `data.yaml` changes. Use
`--all-classes` to evaluate every detector class.

## OOD and unknown-object metrics

The annotated COCO128 boxes are already available after the known validation
script downloads the dataset. Compute OOD novelty and unknown-object metrics
against those boxes from `backend/`:

```powershell
python scripts/run_ood_eval.py
```

The runner uses the actual detector and NOVA-X novelty scorer, matches boxes
to annotations at IoU >= 0.5, and writes `ood_metrics.json`. Unknown-object
recall counts annotated objects missed by the detector as false negatives.
The configured and registered known-class allow-list defines which annotated
classes count as known; all other COCO categories count as unknown.
Unmatched detections count as negatives for OOD ranking. The COCO128 sample is
not held out, so these results are only a sanity check.

## Open-vocabulary metrics

Open-vocabulary accuracy is computed from saved OWLv2 predictions paired with
human-confirmed labels in Human Verification. Those examples are not a held-out
test set and reviewers may have seen the model's suggestions.

For a separate folder-labeled custom evaluation set, add photos under
`data/known_dataset/eval_images/`:

```text
known/                 known-class images
unknown/
  chair/               unknown object photos; folder name is the true label
  laptop/
```

Use one clear subject per image. Unknown labels used for open-vocabulary
evaluation must be in `NOVA_OVD_VOCABULARY`. Start the backend, then run:

```powershell
python scripts/build_eval_labels.py
```

This sends images through the real detection API and writes `ood_labels.csv`
and `ovd_labels.csv` as fallbacks when no ground-truth cache or verified
examples are available. The script defaults to backend port 8001. Refresh the
Evaluation page after running either evaluation script.

# Evaluation images

Add real, held-out test photos here to compute OOD and open-vocabulary metrics:

```text
known/
  person_01.jpg
unknown/
  chair/
    chair_01.jpg
  laptop/
```

Each image should contain one clear primary object. `known/` images should
show known classes; `unknown/<label>/` images should show an object outside
the known classes. The folder name is the true label for open-vocabulary
evaluation and must be in `NOVA_OVD_VOCABULARY`.

With the backend running, execute `python scripts/build_eval_labels.py` from
`backend/`. This supplied folder currently contains no test photos; those
metrics remain unavailable until real labeled photos are added.

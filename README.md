# Apple-tree scaffold recovery (MMA3001 individual project)

Recover the main scaffold (trunk and leaders) of summer apple trees, whose
branches are partly hidden by foliage, from RGB-D images. The work uses the
MMA3001 RGB-D apple-tree dataset.

## Status

Stage 1 is done: indexing, pairing and auditing the data. Later stages
(depth-hole filling, scaffold classification, branch curves and length) will
be added under `apple_scaffold/`.

## Data facts that affect the method

Checked on tree R01N01, 23 Oct 2026; run the audit to confirm across the dataset.

| Item | Finding | Consequence |
|---|---|---|
| Depth PNG | 8-bit greyscale stored as 3 identical channels; `0` = no return; larger = farther | Depth is a rescaled encoding, not mm, so metric 3D needs the scale and camera intrinsics |
| GT PNG | 256 x 256 binary (0/255) | Resize to 640 x 480 with nearest-neighbour, or evaluate at 256 x 256 |
| GT content | Main trunk + two leaders only, drawn as a thick line | Compare centrelines with a pixel tolerance, not just mask IoU |
| File times | `h-m-s`, not zero-padded (`12-41-4`) | Names are parsed with a regular expression |

## Layout

```
apple_scaffold/data.py     filename parsing, frame pairing, loaders, grouped split
scripts/audit_dataset.py   dataset audit: completeness, image properties, split, overlays
tests/test_data.py         pytest tests
```

## Running in Google Colab

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone https://github.com/<your-user>/<this-repo>.git
%cd <this-repo>
!pip install --quiet -r requirements.txt

# audit both dataset folders (edit the paths to where they sit in your Drive)
!python scripts/audit_dataset.py "/content/drive/MyDrive/AppleTree/Dataset (With Summer GT)" \
                                 "/content/drive/MyDrive/AppleTree/Dataset" --out audit

!pytest -v                                   # run the tests
!pdoc --math ./apple_scaffold -o ./docs      # HTML documentation
```

The audit writes `audit/frames.csv` (one row per tree and season, with any
problems), `audit/split.csv` (the saved train/val/test record, grouped by
tree; use `--group-by row` for the stricter row-wise split) and GT overlay
images in `audit/overlays/`.

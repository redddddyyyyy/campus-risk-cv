# Data

Place your source video here before running the pipeline.

The reference run used `IMG_5757.MOV` — a 2-minute 4K clip of a campus crosswalk filmed from a fixed overhead angle.

**To use a different video:** re-calibrate the homography corners with `scripts/homography_picker.py`, then update `configs/zones_img5757.yaml` with the new points.

Video files are gitignored (too large for GitHub). Add the file manually after cloning:

```
data/
└── IMG_5757.MOV    ← place here
```

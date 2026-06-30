# CCE Position Estimation

Classifies each image in a dual-headed colon capsule endoscopy (CCE) recording into one of four intestinal segments: small bowel (SB), ascending colon (AC), transverse colon (TC), or descending colon (DC). Works on raw recordings from the Colon2 pill camera by Medtronic (`.gfd` / `.gvf` format).

Cleanliness and wall-facing estimates are computed first and can optionally be used to exclude uninformative images from the position classifier. A running-mode smoothing step with a configurable image-count window is then applied to produce a robust segment estimate over time. Results are saved as pickle files and displayed in the companion viewer [CCE_viewer.py](https://github.com/kristianhindberg/CCEviewer).

Developed at the UiT Machine Learning Group as part of the [AICE project](https://aiceproject.eu/).

---

## Files

| File | Description |
|---|---|
| `CCE_position.py` | Main script — segment classification, smoothing, and the CLI/import entry point |
| `Position_aux_funcs.py` | Shared auxiliary classes and functions (raw data loading, cleanliness scoring, normalisation stats) |
| `Wall_predictor.py` | Wall-facing image classifier |
| `Cleanliness_predictor.py` | Neural network-based cleanliness classifier |
| `Cleanliness_predictor_colorbased.py` | Color-based cleanliness classifier |
| `CCE_model_position.pth` | Trained position (segment) classifier weights |
| `CCE_model_cleanliness.pth` | Trained cleanliness classifier weights |
| `CCE_model_wall.pth` | Trained wall-facing classifier weights |
| `CCE_model_ICV.pth` | Trained ileocecal valve landmark detector weights |
| `CCE_model_appendix.pth` | Trained appendix landmark detector weights |
| `CCE_model_hepatic_flexure.pth` | Trained hepatic flexure landmark detector weights |
| `CCE_model_splenic_flexure.pth` | Trained splenic flexure landmark detector weights |

---

## Features

- Per-image intestinal segment classification using a trained ResNet model
- Cleanliness and wall-facing pre-filtering
- Optional Markov-style transition probability smoothing
- Uncertainty estimation on the running-mode output
- Batch processing — runs on a single patient folder or all patients in a directory
- Can be triggered directly from the CCE viewer as a background task

---

## Requirements

Python 3.10 or later. Install dependencies via conda:

```bash
conda env create -f environment.yml
conda activate <env_name>
```

Core dependencies: `torch`, `torchvision`, `timm`, `matplotlib`, `numpy`, `Pillow`.

All model weight files must be present in the same directory as the script.

---

## Data format

Each patient folder must contain a five- or six-digit number in its name and the following structure:

```
OUH CHI (10359) 01 Jan 23/
    stream/
        *.gfd
        *.gvf
    stream2/
        *.gfd
        *.gvf
```

Results are written as pickle files directly into the patient folder:

- `Intestine_classifier_results*.pickle` — position estimates
- `PID_XXXXX_WallProb_and_cleanliness.pickle` — cleanliness estimates

---

## Usage

```bash
python CCE_position.py [ARGUMENT]
```

`ARGUMENT` is optional:

- A path to a patient folder — runs inference on that patient only.
- A path to a parent directory — runs inference on all valid patient folders found inside it.
- No argument — scans the current working directory for patient folders.

### Examples

```bash
# Run on a single patient
python CCE_position.py "path/to/OUH CHI (10359) 01 Jan 23"

# Run on all patients in a directory
python CCE_position.py path/to/patient/data

# Run on all patients in the current directory
python CCE_position.py
```

---

## Configuration

The following flags are set near the top of the script and can be edited directly:

| Flag | Default | Description |
|---|---|---|
| `DO_skip_wall_images` | `True` | Exclude wall-facing images from position classification |
| `DO_skip_dirty_images` | `True` | Exclude dirty images from position classification |
| `DO_use_transition_probs` | `True` | Apply Markov-style transition probability smoothing |
| `DO_include_uncertainty` | `True` | Compute and save uncertainty estimates |
| `DO_include_reverse` | `False` | Also run the classifier in reverse temporal order |
| `DO_redo_calc_even_if_done` | `False` | Rerun even if output files already exist |
| `verbose` | `True` | Print progress to the terminal |
| `fd_out_main` | `None` | If set, write output to this folder instead of the patient folder |

---

## Method

### Segment labels

| Label | Segment |
|---|---|
| 0 | Small bowel (SB) |
| 1 | Ascending colon (AC) |
| 2 | Transverse colon (TC) |
| 3 | Descending colon (DC) |

### Smoothing

A fixed image-count window (running mode) is applied to the per-image classifier output. A fixed image-count window is used rather than a fixed time-window because the Colon2 camera dynamically adjusts its frame rate — an image-count window therefore naturally covers more time in low-activity regions and less in high-activity regions.

### Transition probabilities

When `DO_use_transition_probs` is `True`, a Markov-style transition matrix down-weights physiologically unlikely segment jumps (e.g. DC → SB). A small non-zero back-transition probability is retained so that occasional small-bowel misclassifications near the ileocecal valve do not permanently lock the classifier into the colon.

---

## Citation

If you use this software in your research, please cite the following paper (to be updated on publication):

> Hindberg, K. D., et al. (in review). *Title of the position estimation paper*. Journal Name.

```bibtex
@article{hindberg_position,
  author  = {Hindberg, Kristian Dalsbø and others},
  title   = {Title of the position estimation paper},
  journal = {Journal Name},
  year    = {},
  volume  = {},
  pages   = {},
  doi     = {}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

## Author

Kristian Dalsbø Hindberg, UiT — The Arctic University of Norway, 2026

## Related

- [CCEviewer](https://github.com/kristianhindberg/CCEviewer) — CCE viewer with integrated position and cleanliness visualisation

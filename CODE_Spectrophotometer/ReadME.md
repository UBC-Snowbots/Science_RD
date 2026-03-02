# Life Detection Assay — Standard Curve Tools

Scripts for generating standard curves and predicting unknown concentrations
from PySpectrometer2 CSV data.

---

## Files

| File | Purpose |
|------|---------|
| `curve_gen.py` | Main script — builds standard curves and predicts unknowns |
| `fake_data_gen.py` | Generates synthetic CSV files for testing |

---

## Requirements

Install dependencies once before running anything:

```bash
pip install matplotlib numpy pandas scipy
```

---

## curve_gen.py

Has two modes: **build** and **predict**.

### BUILD — create a standard curve from known standards

Run this once when you have your known-concentration CSV files ready.

```bash
python curve_gen.py build \
    --blank blank.csv \
    --standards 0,blank.csv 1,sample_1ppm.csv 2,sample_2ppm.csv 5,sample_5ppm.csv 10,sample_10ppm.csv 20,sample_20ppm.csv \
    --wavelength 525 \
    --unit ppm \
    --name "Life Detection Assay" \
    --model-out curve_model.json
```

**What each argument does:**

| Argument | Required | Description |
|----------|----------|-------------|
| `--blank` | No | Path to blank/reference CSV. Default: `blank.csv` |
| `--standards` | Yes | Space-separated `concentration,filename` pairs |
| `--wavelength` | No | Wavelength in nm to measure at. Default: `525.0` |
| `--unit` | No | Concentration unit label. Default: `ppm` |
| `--name` | No | Assay name shown on plot titles |
| `--model-out` | No | Filename to save the model to. Default: `curve_model.json` |
| `--results-csv` | No | Filename to save per-standard results. Default: `standard_curve_results.csv` |
| `--plot-curve` | No | Filename for standard curve plot. Default: `standard_curve.png` |
| `--plot-spectra` | No | Filename for spectra overlay plot. Default: `spectra_overlay.png` |
| `--no-smooth` | No | Disable Savitzky-Golay smoothing |

**Outputs produced:**

- `curve_model.json` — saved model used for all future predictions
- `standard_curve.png` — plot of absorbance vs concentration with fitted line
- `spectra_overlay.png` — all spectra overlaid with measurement wavelength marked
- `standard_curve_results.csv` — absorbance and back-predicted concentration for each standard

---

### PREDICT — predict concentration of unknown samples

Run this whenever you have a new unknown sample to measure.
Requires a model JSON file produced by the build step.

```bash
# Single unknown
python curve_gen.py predict \
    --sample unknown.csv \
    --model curve_model.json

# Multiple unknowns at once
python curve_gen.py predict \
    --sample unknown1.csv unknown2.csv unknown3.csv \
    --model curve_model.json \
    --results-csv predictions.csv

# With a fresh blank taken on the day (recommended)
python curve_gen.py predict \
    --sample unknown.csv \
    --model curve_model.json \
    --blank todays_blank.csv
```

**What each argument does:**

| Argument | Required | Description |
|----------|----------|-------------|
| `--sample` | Yes | One or more unknown sample CSV files |
| `--model` | No | Model JSON from the build step. Default: `curve_model.json` |
| `--blank` | No | Override the blank stored in the model with a fresh one |
| `--results-csv` | No | Optional path to save predictions as CSV |
| `--no-smooth` | No | Disable Savitzky-Golay smoothing |

---

### Running on a Raspberry Pi over SSH

The script is safe to run headless — it will never try to open a display window.
All output is saved as files you can copy back to your computer.

To copy output files from the Pi to your computer, run this on your computer:

```bash
scp pi@<your-pi-ip>:/path/to/standard_curve.png .
scp pi@<your-pi-ip>:/path/to/curve_model.json .
```

---

## fake_data_gen.py

Generates synthetic CSV files to test the pipeline without real instrument data.

```bash
python fake_data_gen.py
```

Produces: `dark.csv`, `blank.csv`, `sample_1ppm.csv`, `sample_2ppm.csv`,
`sample_5ppm.csv`, `sample_10ppm.csv`, `sample_20ppm.csv`

These can be used directly with `curve_gen.py build` to verify the pipeline is working.

---

## Typical workflow

```
1. Generate test data      →  real samples at various concentrations or python fake_data_gen.py
2. Build curve from tests  →  python curve_gen.py build --standards ...
3. Check output plots      →  open standard_curve.png and spectra_overlay.png
4. Confirm red line sits at trough bottom in spectra_overlay.png
5. Replace test CSVs with real instrument data and repeat steps 2–4
6. Predict unknowns        →  python curve_gen.py predict --sample unknown.csv
```

---

## Interpreting the outputs

**spectra_overlay.png** — Each line is one sample. The red dashed line marks
your measurement wavelength. It should sit at the bottom of the absorption trough.
If it does not, update `--wavelength` to match the actual trough minimum.

**standard_curve.png** — Points are your measured standards. The fitted line
should pass close to all points. R² should be above 0.99 for a good assay.
If it is below this, check for pipetting errors or a drifting light source.

**curve_model.json** — Human-readable file containing the fitted equation,
R², wavelength, and blank path. Safe to open in any text editor.
Keep one per assay type — do not overwrite a working model with a new one
until you are confident in the new curve.

---

## Getting help

```bash
python curve_gen.py build --help
python curve_gen.py predict --help
``` 

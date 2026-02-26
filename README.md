# ROS-SCANNER

## CLI

### Initialize a project

```bash
python survey_recon.py init <image> --out <output_dir> [--page <page_index>]
```

### Run reconstruction

```bash
python survey_recon.py run <output_dir>/project.json \
  [--ocr-engine tesseract|mineru] \
  [--k-linework <int>] \
  [--refine-top-m <int>] \
  [--select-n <int>] \
  [--aoi-scale <float>] \
  [--aoi-threshold <int>] \
  [--pool-layer <name>] \
  [--preview-max-calls <int>] \
  [--page <page_index>]
```

## OCR engine selection

`survey_recon.py` chooses the OCR engine automatically:

- `mineru` on Apple Silicon (`Darwin` + `arm64`/`aarch64`)
- `tesseract` on other platforms

You can override the engine with either:

- `--ocr-engine mineru|tesseract`
- `SURVEY_RECON_OCR_ENGINE=mineru|tesseract`

### MinerU setup for Apple Silicon

Install MinerU from the upstream project:

- https://github.com/opendatalab/MinerU

Ensure the `mineru` CLI is available on `PATH` (or the Python package is installed in the active environment).

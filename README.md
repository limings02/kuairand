# KuaiRand EDA Project

This repository contains exploratory data analysis (EDA) scripts and reporting assets for the KuaiRand dataset.

## Project Structure

- `src/`: EDA modules.
- `run_eda.py`: main entry point for generating analysis outputs.
- `requirements.txt`: Python dependencies.
- `EDA/kuairand_report/`: generated report assets currently in this workspace.
- `EDA/kuairand_interview/`: interview deck source and build scripts.
- `KuaiRand-27K/KuaiRand-27K/load_data_27k.py`: dataset helper script.

## Data Policy

Raw dataset files under:

`KuaiRand-27K/KuaiRand-27K/data/`

are ignored by Git to keep the repository lightweight and compatible with GitHub file size limits.

## Quick Start

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run EDA (replace `DATA_DIR` with your local dataset directory):

```bash
python run_eda.py --data_dir DATA_DIR --out_dir EDA/kuairand_report
```

## Notes

- Line endings are standardized via `.gitattributes`.
- Editor behavior is standardized via `.editorconfig`.
- Temporary files and large raw data are excluded via `.gitignore`.

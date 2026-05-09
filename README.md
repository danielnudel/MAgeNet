# MAgeNet

MAgeNet is a framework for training and applying deep learning models to DNA methylation data for age prediction.  
It provides tools to prepare data, train custom models, and perform predictions on methylation datasets.  
Both single-marker and multi-marker (concatenated) models are supported.

---

## Installation

Clone the repository and install dependencies:

```
git clone https://github.com/danielnudel/MAgeNet.git
cd MAgeNet
pip install -r requirements.txt
```

---

## Data Preparation

Published methylation data lives in `data/published/` as tab-separated `.txt` files, one file per genomic locus (e.g. `chr10:22623318-22623477.txt`).  
Each file has three columns: `age<float>_<sample_tag>`, methylation sequence, read count.

### 1. Convert txt to CSV

```
python3 scripts/txt_to_csv.py data/published/chr6:11044843-11044997.txt -p 1,2,3,4,5,6,7,8,9
```

`-p` selects a subsequence by 1-indexed positions (e.g. `-p 2,3` extracts positions 2 and 3 from each sequence).  
Sequences containing characters other than C or T at the selected positions are discarded.

Output is written to `data/processed/<chr>/<coords>.csv` (e.g. `data/processed/chr6/11044843-11044997.csv`).

### 2. Register and split into train / validation / test

Splits are stratified by age: 70 % train+val / 30 % test, then 70 % train / 30 % validation.

Split assignments are stored permanently in `data/train_test_split.csv`.

**First time — register new samples and create split files:**
```
python3 scripts/split_csv.py data/processed/chr6/11044843-11044997.csv --register
```

**Subsequent runs — only already-registered tags are written to output files:**
```
python3 scripts/split_csv.py data/processed/chr6/11044843-11044997.csv
```

Output files are written alongside the input:
- `data/processed/chr6/11044843-11044997_train.csv`
- `data/processed/chr6/11044843-11044997_validation.csv`
- `data/processed/chr6/11044843-11044997_test.csv`

### 3. (Optional) Concatenate markers for multi-marker training

Run `concat_markers.py` once per split, passing `MARKER=path` pairs:

```
python3 scripts/concat_markers.py \
  ELOVL2_6=data/processed/chr6/11044843-11044997_train.csv \
  C1orf132=data/processed/chr1/207996978-207997090_train.csv \
  -o data/processed/combined/ELOVL2_6_C1orf132_train.csv

python3 scripts/concat_markers.py \
  ELOVL2_6=data/processed/chr6/11044843-11044997_validation.csv \
  C1orf132=data/processed/chr1/207996978-207997090_validation.csv \
  -o data/processed/combined/ELOVL2_6_C1orf132_validation.csv

python3 scripts/concat_markers.py \
  ELOVL2_6=data/processed/chr6/11044843-11044997_test.csv \
  C1orf132=data/processed/chr1/207996978-207997090_test.csv \
  -o data/processed/combined/ELOVL2_6_C1orf132_test.csv
```

Each marker's pattern columns are prefixed with the marker name (e.g. `ELOVL2_6__TTTTTTTTT`) to avoid collisions.  
A `_manifest.json` file is written alongside each output CSV; the augmentation step uses it to augment each marker independently before combining results.

Only samples present in all input markers are kept (inner join on tag).

---

## Training

**Single marker:**
```
python3 train.py \
  -dp  data/processed/chr6/11044843-11044997_train.csv \
  -dpt data/processed/chr6/11044843-11044997_test.csv \
  -dpv data/processed/chr6/11044843-11044997_validation.csv \
  -m   ELOVL2_6
```

**Multi-marker (concatenated):**
```
python3 train.py \
  -dp  data/processed/combined/ELOVL2_6_C1orf132_train.csv \
  -dpt data/processed/combined/ELOVL2_6_C1orf132_test.csv \
  -dpv data/processed/combined/ELOVL2_6_C1orf132_validation.csv \
  -m   ELOVL2_6_C1orf132
```

For concatenated CSVs the model architecture (input size, layer size) is derived automatically from the manifest — no extra flags needed.

| Flag | Description |
|------|-------------|
| `-dp` | Path to training dataset |
| `-dpt` | Path to test dataset |
| `-dpv` | Path to validation dataset |
| `-m` | Marker name: `ELOVL2_6`, `C1orf132`, `FHL2`, `CCDC102B`, or a concatenated combination |
| `-lr` | Learning rate (default: 0.00003) |
| `-bs` | Batch size (default: 128) |
| `-e` | Max epochs (default: 1000) |
| `-d` | Dropout (default: 0.01) |

The trained model is saved to `models/new_predictor_<marker>_<input_size>`.  
Training stops automatically when validation loss stops improving (early stopping).

A quick sanity-check run is also possible with the bundled example data:

```
python3 train.py \
  -dp  example/elovl_example_train.csv \
  -dpt example/elovl_example_test.csv \
  -dpv example/elovl_example_validation.csv \
  -m   ELOVL2_6
```

---

## Prediction

```
python3 predict.py \
  -m  ELOVL2_6 \
  -dp data/processed/chr6/11044843-11044997_test.csv
```

| Flag | Description |
|------|-------------|
| `-m` | Marker name |
| `-dp` | Path to input dataset |
| `-s` | Save results to CSV (default: False) |
| `-o` | Output directory for results CSV (default: `.`) |

---

## Repository Structure

```
MAgeNet/
├── train.py                   # Training script
├── predict.py                 # Prediction script
├── augment_utils.py           # Shared augmentation logic (single- and multi-marker)
├── requirements.txt           # Python dependencies
├── scripts/
│   ├── txt_to_csv.py          # Convert published txt files to CSV
│   ├── split_csv.py           # Split CSV into train/validation/test
│   └── concat_markers.py      # Concatenate per-marker CSVs for multi-marker training
├── data/
│   ├── published/             # Raw txt files per locus
│   ├── processed/             # Converted and split CSVs
│   │   ├── chr<N>/            # Per-chromosome subfolders
│   │   └── combined/          # Concatenated multi-marker CSVs + manifests
│   └── train_test_split.csv   # Canonical split assignment registry
├── models/                    # Saved model weights
├── example/                   # Example datasets for quick testing
└── README.md
```

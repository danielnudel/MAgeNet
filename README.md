# MAgeNet

MAgeNet is a framework for training and applying deep learning models to DNA methylation data for age prediction.  
It provides tools to train custom models and perform predictions on methylation datasets.

---

## Installation

Clone the repository and install dependencies:

```
git clone https://github.com/danielnudel/MAgeNet.git
cd MAgeNet
pip install -r requirements.txt
```


## Training

To train a model, make sure you have all the dependencies from requirements.txt.
You can try a simple example with the provided datasets:

```
python3 train.py \
  -dp example/elovl_example_train.csv \
  -dpt example/elovl_example_test.csv \
  -dpv example/elovl_example_validation.csv \
  -m ELOVL2_6
```

-dp : Path to training dataset

-dpt: Path to test dataset

-dpv: Path to validation dataset

-m : Model name identifier ('ELOVL2_6', 'C1orf132', 'FHL2' or 'CCDC102B')

## Prediction
You can run predictions on a dataset using a trained model:

```
python3 predict.py \
  -m ELOVL2_6 \
  -d example/elovl_example_test.csv
```

-m: Model name identifier

-d: Path to input dataset


## Repository Structure

```
MAgeNet/
│── train.py                 # Training script
│── predict.py               # Prediction script
│── requirements.txt         # Python dependencies
│── example/                 # Example datasets
│   ├── elovl_example_train.csv
│   ├── elovl_example_test.csv
│   └── elovl_example_validation.csv
│── README.md
```

# Classification

## Installation

<!-- ```bash
conda create -n rsna python=3.9
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge pydicom
``` -->

## Data preprocessing for training

```bash
# extracts dcm meta info
python3 preprocess_data.py

# splits train into train-val-test
python3 split_set.py

# extracts npy arrays of size (256, 256) from dcm
python3 extract_npy.py
```


## Model training

### Training

```bash
python3 ./classification/scripts/train.py 2d_11_seresnex50_lr1-4 --config ./res/configs/2d_multichannel.json
python3 ./classification/scripts/train.py 3d_v1 --config ./res/configs/3d.json
```

### Validation

```bash
python3 ./classification/scripts/val.py 2d_11_seresnex50_lr1-4 --split val test
```


## Prediction

from command line
```bash
python3 ./classification/scripts/predict.py 2d_11_seresnex50_lr1-4 ./data/rsna-intracranial-hemorrhage-detection/stage_2_train ./res/preds_2_val.pkl --ids ./data/processed/split_val.txt
python3 ./classification/scripts/predict.py 3d_v2 ./data/rsna-intracranial-hemorrhage-detection/stage_2_train ./res/preds_3_val.pkl --ids ./data/processed/split_val.txt
```

inside python code
```python
from classification.pipeline import ClassificationPipeline

from pathlib import Path
folder = Path("./data/rsna-intracranial-hemorrhage-detection/stage_2_train")
ids = Path("./data/processed/split_val.txt")

model = ClassificationPipeline("2d_11_seresnex50_lr1-4")
ids, preds, patient_ids, patient_preds = model.predict(folder=folder, ids=ids)

model = ClassificationPipeline("3d_v2")
ids, preds, patient_ids, patient_preds = model.predict(folder=folder, ids=ids)
```

## Streamlit

```bash
python3 -m streamlit run ./classification/viz.py
```
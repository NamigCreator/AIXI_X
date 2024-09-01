
# Classification

## Installation

```bash
sudo apt-get install libgl1-mesa-glx xvfb
conda create -n rsna python=3.9
conda activate rsna
conda install -y -c conda-forge numpy
conda install -y -c conda-forge matplotlib pandas scikit-image tqdm
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -y -c conda-forge pydicom
conda install -y -c conda-forge pyvista
conda install -y -c https://conda.anaconda.org/simpleitk SimpleITK
conda install -y -c conda-forge pretrainedmodels
conda install -y -c conda-forge lightning
conda install -y -c conda-forge fire
conda install -y -c conda-forge neptune
conda install -y -c conda-forge vit-pytorch
python3 -m pip install opencv-python
python3 -m pip install streamlit
python3 -m pip install stpyvista
```

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

### Traning of sequential model

Retrieve embeddings from 2D model
```bash
python3 ./classification/scripts/predict.py 2d_v15_rn50_ml_fix ./data/rsna-intracranial-hemorrhage-detection/stage_2_train ./res/preds_train_0.pkl --ids ./data/processed/split_train.txt --get_embeddings
python3 ./classification/scripts/predict.py 2d_v15_rn50_ml_fix ./data/rsna-intracranial-hemorrhage-detection/stage_2_train ./res/preds_val_0.pkl --ids ./data/processed/split_val.txt --get_embeddings
```

Train the sequential model on embeddings from 2D model
```bash
python3 ./classification/scripts/train_embeds.py seq_2 ./res/preds_train_0.pkl ./res/preds_val_0.pkl
```


## Prediction

from command line
```bash
python3 ./classification/scripts/predict.py 2d_11_seresnex50_lr1-4 ./data/rsna-intracranial-hemorrhage-detection/stage_2_train ./res/preds_2_val.pkl --ids ./data/processed/split_val.txt
python3 ./classification/scripts/predict.py 3d_v2 ./data/rsna-intracranial-hemorrhage-detection/stage_2_train ./res/preds_3_val.pkl --ids ./data/processed/split_val.txt
```
with sequential model
```bash
python3 ./classification/scripts/predict.py 2d_v15_rn50_ml_fix ./data/rsna/intracranial-hemorrhage-detection/stage_2_train ./res/preds_test_seq_0.pkl --ids ./data/processed/split_test.txt --seq_model_name seq_2
``` 

inside python code
```python
from classification.pipeline import ClassificationPipeline

from pathlib import Path
folder = Path("./data/rsna-intracranial-hemorrhage-detection/stage_2_train")
ids = Path("./data/processed/split_val.txt")

model = ClassificationPipeline("2d_11_seresnex50_lr1-4")
preds = model.predict(folder=folder, ids=ids)
preds = model.predict(images=dcm_objects)

model = ClassificationPipeline("3d_v2")
preds = model.predict(folder=folder, ids=ids)

model = ClassificationPipeline(model_name="2d_v15_rn50_ml_fix", seq_model_name="seq_2")
preds = model.predict(folder=folder, ids=ids)
```

## Streamlit

Download weights and some case studies from [here](https://drive.google.com/file/d/1SiKU7ZlkMnycoyR1-viPndshfs0LeAKC/view?usp=drive_link),
extract and move checkpoints/ folder into res/ folder


```bash
python3 -m streamlit run ./classification/viz.py
```
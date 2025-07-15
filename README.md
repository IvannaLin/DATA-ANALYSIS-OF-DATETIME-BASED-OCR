# Data Analysis of Datetime-based OCR

## Source Code Access

[GitHub Repository](https://github.com/IvannaLin/DATA-ANALYSIS-OF-DATETIME-BASED-OCR)

## Environment Setup

```bash
# Clean existing env
conda deactivate
conda env remove -n easyocr_ipek -y
conda clean --all -y
pip cache purge

# Create new env
conda create -n easyocr_ipek python=3.10 numpy=1.26 -y
conda activate easyocr_ipek

# Install dependencies
conda install -y scipy pillow scikit-image pyyaml shapely ninja spyder pandas nltk natsort scikit-learn jiwer matplotlib seaborn plyer tensorboard tensorflow
conda install fastai::opencv-python-headless -y
conda install -y -c conda-forge pyclipper python-bidi

# Install EasyOCR
cd path/to/EasyOCR
pip install --use-pep517 --config-settings editable_mode=compat -e .

# Install Intel-optimized PyTorch
python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/xpu
python -m pip install intel-extension-for-pytorch==2.7.10+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

# Verify installations
python -c "import easyocr; print(easyocr.__version__);"
python -c "import easyocr; reader = easyocr.Reader(['en']);"
python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__); [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];"

# Install TensorFlow with Intel extensions
pip install tensorflow==2.15.0
pip install --upgrade intel-extension-for-tensorflow[xpu]
python -c "import intel_extension_for_tensorflow as itex; print(itex.tools.python.env_check.check()); print(itex.__version__)"
```

## Training Custom Model

1. Place your dataset into `trainer/all_data`

```
trainer/
└── all_data/
    ├── 20241025_train/
    │   ├── img1.jpg
    │   ├── ...
    │   └── labels.csv
    └── 20241025_val/
        └── 20241025_val/
            ├── img1.jpg
            ├── ...
            └── labels.csv
```


2. In the YAML configuration file within `trainer/config_files/`, modify the fields as necessary:

```yaml
experiment_name: '20231019_LSTM'
train_data: 'all_data'
valid_data: 'all_data/20231019_val'

# Data processing
select_data: '20231019_train'

# Model Architecture
FeatureExtraction: 'ResNet'
SequenceModeling: 'BiLSTM'

...

```

3. In `trainer/trainer.py`, configure the `configs` list to include the YAML files:

```python
configs = [
    "config_files/delete/20241025_GRU_ResNet_quick_test.yaml",  # Initial testing
    "config_files/20231019_GRU.yaml",
    "config_files/20231019_LSTM.yaml",
    "config_files/20241025_GRU.yaml",
    "config_files/20241025_LSTM.yaml",
    "config_files/20240110_GRU.yaml",
    "config_files/20240110_LSTM.yaml"
]
```

4. Begin training:

```bash
python trainer.py
```

## Visualisation

From the corresponding directory, launch TensorBoard with:

```bash
tensorboard --logdir runs/
```

Access at: `http://localhost:6006`

TensorBoard will display:

* Loss trends
* Accuracy
* Precision, Recall, F1-score
* Training throughput (images/second)
* Total training time (excluding validation)
* Validation time per run

## Citations

This training script builds upon [JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR), which is modified based on:

```bibtex
@inproceedings{baek2019STRcomparisons,
  title={What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis},
  author={Baek, Jeonghun and Kim, Geewook and Lee, Junyeop and Park, Sungrae and Han, Dongyoon and Yun, Sangdoo and Oh, Seong Joon and Lee, Hwalsuk},
  booktitle={International Conference on Computer Vision (ICCV)},
  year={2019}
}
```

For support contact: [ivannalyy@gmail.com](mailto:ivannalyy@gmail.com)

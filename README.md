# DATA ANALYSIS OF DATETIME BASED OCR

## Table of Contents

1. [Source Code Access](#source-code-access)
2. [Environment Setup](#environment-setup)

   * [PaddleOCR Execution](#paddleocr-execution-environment)
   * [EasyOCR Training](#easyocr-training-environment)
3. [Execution Instructions](#execution-instructions)

   * [Running PaddleOCR](#running-paddleocr)
   * [Training Custom Model](#training-custom-model)
4. [Dataset Preparation](#dataset-preparation)
5. [Configuration](#configuration)
6. [Visualization](#visualization)
7. [Citations](#citations)

## Source Code Access

[GitHub Repository]([https://github.com/yourusername/repository](https://github.com/IvannaLin/DATA-ANALYSIS-OF-DATETIME-BASED-OCR))

## Environment Setup

### PaddleOCR Execution Environment

```bash
conda create -n paddleocr2 python=3.8 -y
conda activate paddleocr2
pip install paddlepaddle paddleocr
python -c "import paddle; paddle.utils.run_check()"
python -c "from paddleocr import PaddleOCR; ocr = PaddleOCR(); print('Success!')"
ipython kernel install --user --name=paddleocr2 --display-name "Python 3.8 (PaddleOCR)"
```

### EasyOCR Training Environment

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

## Execution Instructions

### Running PaddleOCR

1. Modify `paddle.py` to specify your image directory:

```python
directories = [
    r'C:\path\to\your\images'  # Change this to your image directory
]
```

2. Run the script:

```bash
python paddle.py
```

### Training Custom Model

1. Prepare dataset (see next section)
2. Configure `trainer.py` to use the desired configurations:

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

3. Start training:

```bash
python trainer.py
```

## Dataset Preparation

### For PaddleOCR

* Place images in specified directory structure
* Script automatically processes all images

### For EasyOCR Training

Modify `easyocr_labels.py`:

```python
def process_csv(input_path, output_dir='output'):
    # Main processing function - no changes needed here

if __name__ == "__main__":
    input_csv = r'C:\path\to\your\timestamps.csv'  # UPDATE THIS PATH
    process_csv(input_csv)
```

## Configuration

Place all YAML configuration files in `config_files/`:

* `20231019_GRU.yaml`: GRU architecture config
* `20231019_LSTM.yaml`: LSTM architecture config

## Visualization

Launch TensorBoard with:

```bash
tensorboard --logdir runs/
```

Access at: `http://localhost:6006`

## Directory structure
```bash
project_root/
│
├── config_files/               # YAML configuration files
│   ├── delete/
│   │   └── 20241025_GRU_ResNet quick test.yaml  # Initial test config
│   │
│   ├── 20231019_GRU.yaml      # GRU config for 20231019 dataset
│   ├── 20231019_LSTM.yaml     # LSTM config for 20231019 dataset
│   ├── 20241025_GRU.yaml      # GRU config for 20241025 dataset  
│   ├── 20241025_LSTM.yaml     # LSTM config for 20241025 dataset
│   ├── 20240110_GRU.yaml      # GRU config for 20240110 dataset
│   └── 20240110_LSTM.yaml     # LSTM config for 20240110 dataset
│
├── all_data/                   # Main data directory
│   ├── 20231019_train/        # Training data for 20231019
│   │   ├── images/
│   │   │   ├── img1.jpg
│   │   │   └── ...
│   │   └── labels.csv         # Format: filename,words
│   │
│   ├── 20231019_val/          # Validation data for 20231019  
│   │   ├── images/
│   │   └── labels.csv
│   │
│   ├── 20241025_train/        # Similar structure for other dates
│   ├── 20241025_val/
│   ├── 20240110_train/
│   └── 20240110_val/
│
├── saved_models/               # Model checkpoints
│   ├── 20231019_GRU_<timestamp>/
│   │   ├── best_accuracy.pth
│   │   ├── best_norm_ED.pth
│   │   ├── iter_*.pth
│   │   ├── opt.txt
│   │   └── log_*.txt
│   └── ... (other experiments)
│
└── runs/                       # TensorBoard logs (now top-level)
    ├── 20231019_GRU_<timestamp>/
```
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

For support contact: \[[ivannalyy@gmail.com](ivannalyy@gmail.com)]

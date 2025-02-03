# Voice Encoder with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) or `venv` ([`+pyenv`](https://github.com/pyenv/pyenv)).

   a. `conda` version:

   ```bash
   # create env
   conda create -n project_env python=PYTHON_VERSION

   # activate env
   conda activate project_env
   ```

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env
   ```

1. Install all required packages

   ```bash
   pip install -r requirements.txt
   ```

2. Install `pre-commit`:
   ```bash
   pre-commit install
   ```

# How To Use

## Model Weights
The trained model weights are located in the `100.pth` file in the `./` directory.

Update the weights path in the following configuration files:
- `src/configs/hifigan.yaml`
- `src/configs/inference.yaml`

Modify the `from_pretrained` field as follows:
```yaml
from_pretrained: [Your_path.pth]
```

## Training the Model
To train the model, execute the following command:
```sh
python3 train.py -cn=hifigan.yaml
```
where `hifigan.yaml` is the configuration file located in `src/configs`. You may also include optional Hydra configuration arguments.

## Dataset Configuration
Update the `dir_path` field in `src/configs/datasets/custom_dir.yaml` to specify the directory where transcriptions are stored.

## Synthesizing Texts
To synthesize your own texts, run:
```sh
python3 syntenizer.py -cn=inferencer.yaml
```
Generated audio files will be saved in the directory containing the transcriptions.

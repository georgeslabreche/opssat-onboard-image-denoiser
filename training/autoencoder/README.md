# Instructions
- Training images are not provided (yet).
- Instructions are written for Windows with Conda.

## Installing
```powershell
conda create -n denoiser
conda activate denoiser
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
conda install -c nvidia cuda-nvcc=11.3
pip install -r requirements.txt
```

## Training
```powershell
conda activate denoiser
python 03_train_autoencoder.py
```

## Todo
There's a lot of cleaning up left todo:
1. Remove python files that are no longer relevent (00_find_bad_jpegs.py and 02_merge_bands.py).
2. Update 01_fetch_imagery.py to fetch OPS-SAT-1 training imagery.
3. Write a script to prepare all the training data (noised and patched).

# Save the Knees

## Installation Instructions

Clone the repository

Create the virutal env
```
pyenv virtualenv 3.10.6 save-the-knees-env

pyenv local save-the-knees-env

pip install -r requirements.txt
```

## Train on your own machine (Mac / Windows / Linux)

The training data lives as preprocessed blobs in Google Cloud Storage — nobody
needs the 600 GB raw DICOM. One command builds the env for YOUR platform
(NVIDIA CUDA on Linux/WSL2, Apple Metal on macOS) and pulls the ~16 GB of blobs:

```
gcloud auth login          # once, with your account on the savetheknees project
bash setup_training.sh
```

It prints the canonical training commands when done. Windows + NVIDIA: run it
inside WSL2 (native Windows TensorFlow is CPU-only). Trained checkpoints are
pushed back to `gs://knees-models` under your own name — promotion happens only
via the referee on the 58 gold studies (`models/evaluate_labels.py`).

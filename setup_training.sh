#!/usr/bin/env bash
# One command to train SaveTheKnees models on YOUR machine, whatever it is.
#
#   bash setup_training.sh          # env + download the training blobs (~16 GB)
#
# Needs: python 3.10+, and gcloud authed on the 'savetheknees' project
# (https://cloud.google.com/sdk/docs/install then `gcloud auth login`).
# Training reads the preprocessed tensor blobs from GCS -- nobody needs the
# 600 GB raw DICOM. Trained checkpoints are pushed back to gs://knees-models
# under NEW names; promotion still goes through the referee on the 58 gold.
set -euo pipefail
cd "$(dirname "$0")"

command -v gcloud >/dev/null || { echo "install the gcloud CLI first: https://cloud.google.com/sdk/docs/install"; exit 1; }
gcloud storage ls gs://knees-models >/dev/null 2>&1 || { echo "run: gcloud auth login   (with your account on the savetheknees project)"; exit 1; }

# --- python env (bin/ on mac+linux, Scripts/ on windows git-bash) -----------
PY=python3; command -v python3 >/dev/null || PY=python
[ -d save-the-knees-env ] || $PY -m venv save-the-knees-env
VPY=save-the-knees-env/bin/python; [ -f "$VPY" ] || VPY=save-the-knees-env/Scripts/python.exe
"$VPY" -m pip install -q -r requirements-train.txt
echo "GPU check:"; "$VPY" -c "import tensorflow as tf; print(' ', tf.config.list_physical_devices('GPU') or 'no GPU visible -> CPU (see requirements-train.txt header)')"

case "$(uname -sr)" in
  *[Mm]icrosoft*) : ;;  # WSL2: CUDA wheels see the Windows NVIDIA GPU, all good
  MINGW*|MSYS*|CYGWIN*) echo "NOTE: native Windows = CPU-only TF. For NVIDIA GPU training, run this same script inside WSL2.";;
esac
if command -v nvidia-smi >/dev/null && nvidia-smi -L 2>/dev/null | grep -q "RTX 50"; then
  echo "NOTE: RTX 50-series (sm_120) needs the NGC container: docker compose run train"
fi

# --- the training blobs (resumable; only downloads what's missing) ----------
mkdir -p data models
gcloud storage rsync -r -q gs://knees-images/meta data/meta
gcloud storage cp -q "gs://knees-images/*.csv" data/ 2>/dev/null || true
gcloud storage cp -n -q "gs://knees-models/*.keras" "gs://knees-models/*.joblib" models/   # -n: skip files already present
echo "tensor blobs (~15 GB, resumable -- rerun this script if interrupted):"
gcloud storage rsync -r gs://knees-images/tensor_cache data/tensor_cache

cat <<'CMDS'

Ready. Canonical runs (pick a NEW --seed, don't overwrite incumbents):
  save-the-knees-env/*/python -m models.train_model  --augment --epochs 100 --patience 15 --seed N --checkpoint models/knee_findings_yourname.keras
  save-the-knees-env/*/python -m models.train_model  --multi-plane --augment --epochs 100 --patience 15 --seed N --checkpoint models/knee_multiplane_yourname.keras
  save-the-knees-env/*/python -m models.train_fusion --augment --mixup --warm-start models/knee_multiplane.keras --epochs 100 --patience 15 --seed N --checkpoint models/knee_fusion_yourname.keras --gold-out data/meta/fusion_yourname_gold.csv
  save-the-knees-env/*/python -m models.report_model --epochs 40 --patience 5 --seed N

Then share your result:
  gcloud storage cp models/*yourname*.keras gs://knees-models/
  gcloud storage rsync -r data/meta gs://knees-images/meta
CMDS

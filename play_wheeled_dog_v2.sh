#!/usr/bin/env bash
# Play checkpoint in Isaac Sim GUI (keyboard teleop).
# WebRTC: GUI=0 LIVESTREAM=2 bash play_wheeled_dog_v2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "${CONDA_ENV:-isaaclab51}"

export ZSH_VERSION=""
export OMNI_KIT_ACCEPT_EULA=YES
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
export PYTHONUNBUFFERED=1

ISAACLAB_ROOT="${ISAACLAB_ROOT:-$ROOT/IsaacLab}"
set +u
# shellcheck source=/dev/null
source "$ISAACLAB_ROOT/_isaac_sim/setup_conda_env.sh"
set -euo pipefail

cd "$ROOT/rl_training"
TASK="${TASK:-Rough-Wheeled-Dog-V2-v0}"
NUM_ENVS="${NUM_ENVS:-1}"
GUI="${GUI:-1}"
KEYBOARD="${KEYBOARD:-1}"

EXP_DIR="logs/rsl_rl/wheeled_dog_v2_rough"
if [[ -z "${RUN_DIR:-}" ]]; then
  RUN_DIR=$(ls -td "$EXP_DIR"/*/ 2>/dev/null | head -1)
fi
RUN_NAME=$(basename "${RUN_DIR%/}")
CKPT="${CKPT:-}"

CMD=(python -u scripts/reinforcement_learning/rsl_rl/play.py
  --task="$TASK"
  --num_envs="$NUM_ENVS"
  --load_run="$RUN_NAME"
)
if [[ -n "$CKPT" ]]; then
  CMD+=(--checkpoint "$CKPT")
fi
if [[ "$KEYBOARD" == "1" ]]; then
  CMD=(python -u scripts/reinforcement_learning/rsl_rl/play.py
    --task="$TASK" --num_envs=1 --load_run="$RUN_NAME" --keyboard)
  [[ -n "$CKPT" ]] && CMD+=(--checkpoint "$CKPT")
fi
if [[ "$GUI" != "1" ]]; then
  CMD+=(--livestream "${LIVESTREAM:-2}")
  echo "[play] WebRTC mode — connect client to 127.0.0.1"
else
  echo "[play] native Isaac Sim GUI on DISPLAY=$DISPLAY"
  echo "[play] keys: arrows move | Z/X yaw | L zero cmd | R reset robot"
fi

echo "[play] run=$RUN_NAME ckpt=${CKPT:-latest}"
echo "[play] ${CMD[*]}"
exec "${CMD[@]}"

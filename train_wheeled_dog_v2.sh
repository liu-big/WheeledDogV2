#!/usr/bin/env bash
# Train WheeledDogV2 rough/stair policy (Isaac Lab).
# Default: GUI on DISPLAY=:0, 4096 envs.
# Headless: GUI=0 bash train_wheeled_dog_v2.sh
# Resume:   RESUME=1 LOAD_RUN=<run> CHECKPOINT=model_xxxx.pt bash train_wheeled_dog_v2.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"

# --- conda (override CONDA_SH if needed) ---
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate "${CONDA_ENV:-isaaclab51}"

export ZSH_VERSION=""
export OMNI_KIT_ACCEPT_EULA=YES
export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-$HOME/.Xauthority}"
export PYTHONUNBUFFERED=1

# Isaac Lab kit env (override ISAACLAB_ROOT)
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$ROOT/IsaacLab}"
if [[ ! -f "$ISAACLAB_ROOT/_isaac_sim/setup_conda_env.sh" ]]; then
  echo "[train] missing $ISAACLAB_ROOT/_isaac_sim/setup_conda_env.sh"
  echo "        set ISAACLAB_ROOT to your Isaac Lab install"
  exit 1
fi
set +u
# shellcheck source=/dev/null
source "$ISAACLAB_ROOT/_isaac_sim/setup_conda_env.sh"
set -euo pipefail

cd "$ROOT/rl_training"

TASK="${TASK:-Rough-Wheeled-Dog-V2-v0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-}"
GUI="${GUI:-1}"
RESUME="${RESUME:-0}"
LOAD_RUN="${LOAD_RUN:-}"
CHECKPOINT="${CHECKPOINT:-}"
EXTRA_ARGS=("$@")

CMD=(python -u scripts/reinforcement_learning/rsl_rl/train.py --task="$TASK" --num_envs="$NUM_ENVS")
if [[ -n "$MAX_ITERS" ]]; then
  CMD+=(--max_iterations="$MAX_ITERS")
fi
if [[ "$RESUME" == "1" ]]; then
  CMD+=(--resume)
  [[ -n "$LOAD_RUN" ]] && CMD+=(--load_run="$LOAD_RUN")
  [[ -n "$CHECKPOINT" ]] && CMD+=(--checkpoint="$CHECKPOINT")
  echo "[train] resume load_run=${LOAD_RUN:-auto} checkpoint=${CHECKPOINT:-latest}"
fi
if [[ "$GUI" != "1" ]]; then
  if [[ -n "${LIVESTREAM:-}" ]]; then
    CMD+=(--livestream "$LIVESTREAM")
    echo "[train] WebRTC livestream=$LIVESTREAM"
  else
    CMD+=(--headless)
    echo "[train] headless"
  fi
else
  echo "[train] Isaac Sim GUI on DISPLAY=$DISPLAY  num_envs=$NUM_ENVS"
fi
CMD+=("${EXTRA_ARGS[@]}")

echo "[train] cwd=$(pwd)"
echo "[train] ${CMD[*]}"
exec "${CMD[@]}"

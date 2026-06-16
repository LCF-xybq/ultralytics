#!/bin/bash
# Sequentially run all 9 ablation training scripts.
# Each script's results land in ../../runs_ablations/<name>/.
# Activate the conda env first: `conda activate yolo`.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

for script in \
    hp_only_v8.py cc_only_v8.py full_v8.py \
    hp_only_11.py cc_only_11.py full_11.py \
    hp_only_26.py cc_only_26.py full_26.py
do
    echo "===== Running $script ====="
    python "$script" || echo "===== FAILED: $script ====="
done

echo "All 9 scripts attempted."

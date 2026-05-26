#!/bin/bash
set -euo pipefail

# Only run in remote Claude Code environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

echo "=== Session start: installing dependencies ==="

# Skip Windows-only (pywin32) and Linux-unbuildable legacy packages (fvcore, iopath, PyTurboJPEG)
grep -vE "^(pywin32|fvcore|iopath|PyTurboJPEG)" "$CLAUDE_PROJECT_DIR/requirements.txt" \
  | pip install -q --ignore-installed -r /dev/stdin

echo "=== Verifying core imports ==="
python -c "
import torch
from models.Vision_Transformer import ViT_Hierarchical
from models.Motion_Encoder import MotionEncoder
from models.Cross_Attention_Module import CrossAttentionModule
from models.Unified_Module import EnsembleModel
print('CUDA:', torch.cuda.is_available())
print('All imports OK')
"

echo "=== Session ready ==="

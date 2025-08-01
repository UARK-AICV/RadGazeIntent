#!/bin/bash
# Add all changes
git add .

# Create commit message with timestamp
commit_message="Auto commit: $(date '+%Y-%m-%d %H:%M:%S')"

# Commit changes
git commit -m "$commit_message"

# Push to remote repository
git push

echo "Changes committed and pushed successfully"

CUDA_VISIBLE_DEVICES=2 python train.py --hparams configs/train_for_real_reflacx_s2.json
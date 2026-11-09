#!/usr/bin/env bash
# Task 3.5 - the experiment grid. Each run differs from the reference run
# (mnv2_finetune) in exactly one meaningful setting.
set -euo pipefail
cd "$(dirname "$0")/.."
export TF_CPP_MIN_LOG_LEVEL=2
T="python scripts/train.py --experiment VisionGuard"
case "${1:-all}" in
  main|all)
    $T --run-name cnn_baseline          --model cnn --lr 1e-3 --epochs 30 --patience 6 --dropout 0.4 \
       --description "Baseline CNN from scratch" --save-as cnn_baseline
    $T --run-name mnv2_finetune         --model transfer --backbone mobilenetv2 --lr 1e-3 --ft-lr 1e-5 \
       --epochs 15 --ft-epochs 15 --description "Reference: MobileNetV2 frozen head then last block fine-tuned"
    $T --run-name mnv2_head_only        --model transfer --backbone mobilenetv2 --lr 1e-3 --epochs 15 --ft-epochs 0 \
       --description "Change: no fine-tuning stage (backbone fully frozen)"
    $T --run-name mnv2_finetune_noaug   --model transfer --backbone mobilenetv2 --lr 1e-3 --ft-lr 1e-5 \
       --epochs 15 --ft-epochs 15 --no-augment --description "Change: augmentation off"
    $T --run-name mnv2_finetune_lr1e-4  --model transfer --backbone mobilenetv2 --lr 1e-4 --ft-lr 1e-5 \
       --epochs 15 --ft-epochs 15 --description "Change: head learning rate 1e-4"
    $T --run-name mnv2_finetune_img160  --model transfer --backbone mobilenetv2 --img-size 160 --lr 1e-3 --ft-lr 1e-5 \
       --epochs 15 --ft-epochs 15 --description "Change: image size 160"
    $T --run-name mnv2_finetune_unfreeze40 --model transfer --backbone mobilenetv2 --lr 1e-3 --ft-lr 1e-5 \
       --epochs 15 --ft-epochs 15 --unfreeze 40 --description "Change: unfreeze last 40 backbone layers"
    $T --run-name mnv2_finetune_autolog --model transfer --backbone mobilenetv2 --lr 1e-3 --ft-lr 1e-5 \
       --epochs 15 --ft-epochs 15 --logging autolog --description "Reference config with mlflow.tensorflow.autolog"
    ;;&
  extra|all)
    $T --run-name resnet50_finetune     --model transfer --backbone resnet50 --lr 1e-3 --ft-lr 1e-5 \
       --epochs 15 --ft-epochs 10 --description "Change: ResNet50 backbone"
    ;;
esac

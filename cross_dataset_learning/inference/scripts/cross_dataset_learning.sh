#!/bin/bash

families=("LLM4TS" "TSFMs")
scales=("Tiny" "Small" "Base" "Large")

for family in "${families[@]}"; do
  for scale in "${scales[@]}"; do
    echo ">>> Running: --family $family --scales $scale"
    accelerate launch \
      --config_file main/cross_dataset_learning/inference/scripts/accelerate_config.yaml \
      main/cross_dataset_learning/inference/run.py \
      --family "$family" \
      --scales "$scale"
  done
done
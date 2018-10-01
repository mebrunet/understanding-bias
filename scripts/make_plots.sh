#!/bin/bash

SAVE_DIR=results/figures
mkdir -p $SAVE_DIR

for target in $(ls -d results/perturbations/*); do
  python scripts/make_plots.py $target $SAVE_DIR
done

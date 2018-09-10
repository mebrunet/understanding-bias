#!/bin/bash

# Setup repository after cloning

BASE_DIR="$(dirname $0)/.."

# Initialize and update submodules
echo Initializing git submodules...
git submodule init && git submodule update

# Setup python environment
echo Setting up python environment...
virtualenv -p python3 venv || python3 -m venv venv
venv/bin/pip install -r requirements.txt

# Setup julia environment
echo Setting up julia environment...
julia --project -e "import Pkg; Pkg.instantiate()"

# Make Wiki corpus
echo Downloading and building Simple Wikipedia corpus...
scripts/init_corpora.sh

# Build GloVe and train a Toy Embedding
echo Building GloVe and training a toy embedding...
make -C "$BASE_DIR/GloVe"
scripts/embed.sh scripts/toy_embed.config

# Evaluate annalogy performance
echo Evaluating analogy performace...
scripts/analogy.sh embeddings/vectors-C0-V20-W8-D25-R0.05-E15-S1.bin

# Run tests
echo Running tests
./test

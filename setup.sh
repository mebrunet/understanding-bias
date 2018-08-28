#!/bin/bash

# Setup repository after cloning

BASE_DIR="$(dirname $0)/.."

# Initialize and update submodules
echo Initializing git submodules...
git submodule init && git submodule update

# Setup python environment
echo Setting up python environment...
virtualenv -p python3 venv
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

# Run tests
echo Running tests
./test

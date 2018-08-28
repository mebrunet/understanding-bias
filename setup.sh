#!/bin/bash

# Setup repository after cloning

# Initialize and update submodules
echo Initializing git submodules...
git submodule init && git submodule update

# Setup python environment
echo Setting up python environment...
virtualenv -p python3 venv
venv/bin/pip install -r requirements.txt

# Setup julia environment
julia --project -e "import Pkg; Pkg.instantiate()"

# Make Wiki corpus
echo Downloading and building Simple Wikipedia corpus...
scripts/init_corpora.sh

# Run tests
./test

#!/bin/bash

BASE_DIR="$(dirname $0)/.."

EMBED_BIN=$PWD/$1 # Get embedding's abs path
EMBED_TXT=${EMBED_BIN%.*}.txt.tmp # Create name for txt vectors

cd $BASE_DIR/GloVe # change directory to GloVe

VOCAB=`julia --project -e "
include(\"../src/GloVe.jl\");
M=GloVe.load_model(\"$EMBED_BIN\");
GloVe.save_text_vectors(\"$EMBED_TXT\", M.W, M.ivocab);
print(M.vocab_path)
"`

../venv/bin/python eval/python/evaluate.py --vocab_file $VOCAB --vectors_file $EMBED_TXT

rm $EMBED_TXT

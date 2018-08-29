#!/bin/bash

# Build an extremely small corpus and cooc matrix for rapid testing

BASE_DIR="$(dirname $0)/.."

BUILD_DIR=$BASE_DIR/GloVe/build
RESULTS_DIR=$BASE_DIR/embeddings
TEST_DIR=$BASE_DIR/tests

CORPUS_FILE=$BASE_DIR/corpora/simplewikiselect.txt
VOCAB_FILE=$RESULTS_DIR/vocab-C0-V20.txt

TEST_CORPUS=$TEST_DIR/test_corpus.txt
TEST_COOC=$TEST_DIR/test_cooc.bin

head -n 25 $CORPUS_FILE > $TEST_CORPUS

$BUILD_DIR/cooccur -vocab-file $VOCAB_FILE -verbose 2 -window-size 8 -overflow-file $RESULTS_DIR/overflow < $TEST_CORPUS > $TEST_COOC

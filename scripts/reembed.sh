#!/bin/bash

set -e

BASE_DIR="$(dirname $0)/.."
BUILD_DIR=$BASE_DIR/GloVe/build  # GloVe binaries are here

cd $BASE_DIR

TARGET=$1
PERT_DIR=$2
EMBEDDING_DIR=$3
MEMORY=${4:-"6"}
NUM_THREADS=${5:-"6"}
VERBOSE=${6:-"2"}


RESULTS_DIR=$PERT_DIR/$TARGET
VOCAB_FILE=$EMBEDDING_DIR/vocab-${TARGET%%-W*}.txt
BASE_COOC_FILE=$EMBEDDING_DIR/cooc-${TARGET%%-D*}.bin
CONFIG_FILE=$RESULTS_DIR/config.txt

TMP_COOC_FILE=$RESULTS_DIR/cooc-tmp.bin
TEMP_FILE=$RESULTS_DIR/temp.bin
TMP_SHUF_FILE=$RESULTS_DIR/shuf-tmp.bin

echo "VOCAB: $VOCAB_FILE"
echo "BASE COOCS: $BASE_COOC_FILE"
echo "CONFIG_FILE: $CONFIG_FILE"

source $CONFIG_FILE
echo Loaded Configuration:
echo CORPUS_ID = $CORPUS_ID
echo VOCAB_MIN_COUNT = $VOCAB_MIN_COUNT
echo WINDOW_SIZE = $WINDOW_SIZE
echo VECTOR_SIZE = $VECTOR_SIZE
echo ETA = $ETA
echo MAX_ITER = $MAX_ITER
echo


add_pert() {
  PERT_FILE=$1
  echo "Adding $PERT_FILE"
  julia --project $BASE_DIR/src/add_perturbation.jl $BASE_COOC_FILE $PERT_FILE $TMP_COOC_FILE $VOCAB_FILE
}


for pert_path in $(ls $PERT_DIR/$TARGET/pert-*); do
  PERT_FILENAME=$(basename $pert_path)
  add_pert $pert_path
  TMP=${PERT_FILENAME##pert-}
  TMP=${TMP%%.bin}
  for SEED in $(seq 1 5); do
      SAVE_FILE=$RESULTS_DIR/vectors-${TMP}_$SEED  # Don't include .bin

      echo "Reshuffling with seed $SEED"
      $BUILD_DIR/shuffle -memory $MEMORY -verbose $VERBOSE -temp-file $TEMP_FILE -seed $SEED < $TMP_COOC_FILE > $TMP_SHUF_FILE

      echo "Retraining with seed $SEED"
      $BUILD_DIR/glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $TMP_SHUF_FILE -iter $MAX_ITER -checkpoint-every 0 -vector-size $VECTOR_SIZE -binary 1 -vocab-file $VOCAB_FILE -eta $ETA -verbose $VERBOSE -seed $SEED

      # Clean up
      rm $TMP_SHUF_FILE
  done
  rm $TMP_COOC_FILE
done

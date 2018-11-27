#!/bin/bash

EBED_DIR=/scratch/gobi1/mebrunet/w2v
CODE_DIR=/h/mebrunet/Code/understanding-bias
TEST_WORDS_PATH=$CODE_DIR/scripts/question-words.txt

echo "scenario,seed,num_docs,accuracy,effect_size"

for f in $(ls $EBED_DIR/*.w2v); do
  python $CODE_DIR/scripts/eval_w2v.py $f $TEST_WORDS_PATH
done

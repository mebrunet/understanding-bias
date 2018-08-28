#!/bin/bash

# Dowload simple english wikipedia and create a corpus with it

BASE_DIR="$(dirname $0)/.."

FILENAME="simplewiki-20171103-pages-articles-multistream.xml.bz2"
URL="http://www.cs.toronto.edu/~mebrunet/$FILENAME"
echo $URL


if [ ! -e "$BASE_DIR/corpora/$FILENAME" ]; then
  if hash wget 2>/dev/null; then
    wget -O $BASE_DIR/corpora/$FILENAME $URL
  else
    curl -o $BASE_DIR/corpora/$FILENAME $URL
  fi
fi

venv/bin/python $BASE_DIR/scripts/make_wiki_corpus.py

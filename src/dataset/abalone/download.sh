#!/usr/bin/env bash

mkdir -p data/abalone
cd data/abalone || (echo "data/abalone does not exist. Exiting."; exit 1)

if [ -f abalone.data ]; then
    echo "abalone.data already exists. Skip downloading."
else
    echo "Downloading abalone.data..."
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data
fi

if [ -f abalone.names ]; then
    echo "abalone.names already exists. Skip downloading."
else
    echo "Downloading abalone.names..."
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names
fi

# Create a raw directory and move the files there
mkdir -p raw
mv abalone.data abalone.names raw/

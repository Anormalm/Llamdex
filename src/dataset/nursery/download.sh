#!/usr/bin/env bash

mkdir -p data/nursery
cd data/nursery || (echo "data/nursery does not exist. Exiting."; exit 1)

if [ -f nursery.data ]; then
    echo "nursery.data already exists. Skip downloading."
else
    echo "Downloading nursery.data..."
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data
fi

if [ -f nursery.names ]; then
    echo "nursery.names already exists. Skip downloading."
else
    echo "Downloading nursery.names..."
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.names
fi

# Create a raw directory and move the files there
mkdir -p raw
mv nursery.data nursery.names raw/



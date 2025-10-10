#!/usr/bin/env bash

mkdir -p data/titanic/raw
cd data/titanic/raw || (echo "Failed to create or access data/titanic/raw directory. Exiting."; exit 1)

if [ -f titanic.csv ]; then
    echo "titanic.csv already exists. Skip downloading."
else
    echo "Downloading titanic.csv..."
    wget https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv
fi

echo "Titanic dataset has been downloaded to the 'raw' directory."
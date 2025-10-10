#!/usr/bin/env bash

mkdir -p data/wine_quality
cd data/wine_quality || (echo "data/wine_quality does not exist. Exiting."; exit 1)

if [ -f wine+quality.zip ]; then
    echo "wine+quality.zip already exists. Skip downloading."
else
    echo "Downloading wine+quality.zip..."
    wget https://archive.ics.uci.edu/static/public/186/wine+quality.zip
fi

unzip wine+quality.zip -d raw





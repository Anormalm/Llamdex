#!/usr/bin/env bash

mkdir -p data/android
cd data/android || (echo "data/android does not exist. Exiting."; exit 1)

if [ -f android.zip ]; then
    echo "android.zip already exists. Skip downloading."
else
    echo "Downloading android.zip..."
    wget https://archive.ics.uci.edu/static/public/722/naticusdroid+android+permissions+dataset.zip -O android.zip
fi

unzip android.zip -d raw
mv raw/data.csv raw/android.csv
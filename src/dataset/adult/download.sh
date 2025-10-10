#!/usr/bin/env bash

mkdir -p data/adult
cd data/adult || (echo "data/adult does not exist. Exiting."; exit 1)

if [ -f adult.zip ]; then
    echo "adult.zip already exists. Skip downloading."
else
    echo "Downloading adult.zip..."
    wget https://archive.ics.uci.edu/static/public/2/adult.zip
fi

unzip adult.zip -d raw





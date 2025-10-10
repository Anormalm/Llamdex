#!/usr/bin/env bash

mkdir -p data/bank_marketing
cd data/bank_marketing || (echo "data/bank_marketing does not exist. Exiting."; exit 1)

if [ -f bank.zip ]; then
    echo "bank.zip already exists. Skip downloading."
else
    echo "Downloading bank.zip..."
    wget https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip
fi

unzip bank.zip -d raw
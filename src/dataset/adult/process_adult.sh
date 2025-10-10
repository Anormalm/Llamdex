#!/usr/bin/env bash

python src/preprocess/clean/clean_adult.py
python src/preprocess/syn/syn_adult.py
python src/preprocess/expert/adult_mlp.py
python src/preprocess/gentext/gentext_adult.py


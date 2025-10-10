#!/usr/bin/env bash

python src/preprocess/clean/clean_wine_quality.py
python src/preprocess/syn/syn_wine_quality.py
python src/preprocess/expert/wine_quality_mlp.py
python src/preprocess/gentext/gentext_wine_quality.py
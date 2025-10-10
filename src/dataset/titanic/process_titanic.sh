#!/usr/bin/env bash

python src/preprocess/clean/clean_titanic.py
python src/preprocess/syn/syn_titanic.py
python src/preprocess/expert/titanic_mlp.py
python src/preprocess/gentext/gentext_titanic.py


#!/usr/bin/env bash

mkdir -p log/synthetic

algo="pate-gan"
for eps in 0.5 1 2 3 4 5 6; do
#  # adult
#  dataset="adult"
#  label='salary>50K'
#  delta=1e-5
#  savepath="data/${dataset}/dpsyn/eps${eps}/${algo}/"
#  mkdir -p ${savepath}
#  CUDA_VISIBLE_DEVICES=0 python baseline/private-data-generation/evaluate.py --target-variable ${label} --train-data-path data/${dataset}/clean/${dataset}_train_onehot.csv --test-data-path data/${dataset}/clean/${dataset}_test_onehot.csv --normalize-data ${algo} --enable-privacy --target-epsilon=${eps} --target-delta=${delta} --save-synthetic --output-data-path ${savepath} > log/synthetic/${algo}-${dataset}-eps${eps}.log &
#
#  # titanic
#  dataset="titanic"
#  label="Survived"
#  delta=1e-3
#  savepath="data/${dataset}/dpsyn/eps${eps}/${algo}/"
#  mkdir -p ${savepath}
#  CUDA_VISIBLE_DEVICES=1 python baseline/private-data-generation/evaluate.py --target-variable ${label} --train-data-path data/${dataset}/clean/${dataset}_train_onehot.csv --test-data-path data/${dataset}/clean/${dataset}_test_onehot.csv --normalize-data ${algo} --enable-privacy --target-epsilon=${eps} --target-delta=${delta} --save-synthetic --output-data-path ${savepath} > log/synthetic/${algo}-${dataset}-eps${eps}.log &
#  wait

#  # wine_quality
#  dataset="wine_quality"
#  label="quality"
#  delta=1e-3
#  savepath="data/${dataset}/dpsyn/eps${eps}/${algo}/"
#  mkdir -p ${savepath}
#  CUDA_VISIBLE_DEVICES=2 python baseline/private-data-generation/evaluate.py --target-variable ${label} --train-data-path data/${dataset}/clean/${dataset}_train_onehot.csv --test-data-path data/${dataset}/clean/${dataset}_test_onehot.csv --normalize-data ${algo} --enable-privacy --target-epsilon=${eps} --target-delta=${delta} --save-synthetic --output-data-path ${savepath} > log/synthetic/${algo}-${dataset}-eps${eps}.log

# bank_marketing
  dataset="bank_marketing"
  label="subscribed"
  delta=1e-4
  savepath="data/${dataset}/dpsyn/eps${eps}/${algo}/"
  mkdir -p ${savepath}
  CUDA_VISIBLE_DEVICES=0 python baseline/private-data-generation/evaluate.py --target-variable ${label} --train-data-path data/${dataset}/clean/${dataset}_train_onehot.csv --test-data-path data/${dataset}/clean/${dataset}_test_onehot.csv --normalize-data ${algo} --enable-privacy --target-epsilon=${eps} --target-delta=${delta} --save-synthetic --output-data-path ${savepath} > log/synthetic/${algo}-${dataset}-eps${eps}.log
#
#  # abalone
#  dataset="abalone"
#  label="Age_Group"
#  delta=1e-4
#  savepath="data/${dataset}/dpsyn/eps${eps}/${algo}/"
#  mkdir -p ${savepath}
#  CUDA_VISIBLE_DEVICES=3 python baseline/private-data-generation/evaluate.py --target-variable ${label} --train-data-path data/${dataset}/clean/${dataset}_train_onehot.csv --test-data-path data/${dataset}/clean/${dataset}_test_onehot.csv --normalize-data ${algo} --enable-privacy --target-epsilon=${eps} --target-delta=${delta} --save-synthetic --output-data-path ${savepath} > log/synthetic/${algo}-${dataset}-eps${eps}.log
done


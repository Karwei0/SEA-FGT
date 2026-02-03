#!/bin/bash
GPU=9
EPOCHS=3
LR=0.001
LAYERS=3
IRADJ="cosine"
DMODEL=256
BS=160
DATASET="MSL"
# w/o CCE
# python run_simple.py \
#   --n_layers=$LAYERS \
#   --train_epochs=$EPOCHS \
#   --Iradj=$IRADJ \
#   --learning_rate=$LR \
#   --gpu=$GPU \
#   --n_head=1 \
#   --T=96 \
#   --topk=2 \
#   --batch_size=$BS \
#   --bin_size=10 \
#   --step=24 \
#   --step_eq_win_size=0 \
#   --d_model=$DMODEL \
#   --dataset=$DATASET \
#   --use_sea=1 \
#   --use_cce=0 \
#   --use_fgt=1 \
#   --exp_name=${DATASET}_ABL_NO_CCE

# w/o SEA
python run_simple.py \
  --n_layers=$LAYERS \
  --train_epochs=$EPOCHS \
  --Iradj=$IRADJ \
  --learning_rate=$LR \
  --gpu=$GPU \
  --n_head=1 \
  --T=96 \
  --topk=2 \
  --batch_size=$BS \
  --bin_size=10 \
  --step=24 \
  --step_eq_win_size=0 \
  --d_model=$DMODEL \
  --dataset=$DATASET \
  --use_sea=0 \
  --use_cce=1 \
  --use_fgt=1 \
  --exp_name=${DATASET}_ABL_NO_SEA

# w/o FGT
# python run_simple.py \
#   --n_layers=$LAYERS \
#   --train_epochs=$EPOCHS \
#   --Iradj=$IRADJ \
#   --learning_rate=$LR \
#   --gpu=$GPU \
#   --n_head=1 \
#   --T=96 \
#   --topk=2 \
#   --batch_size=$BS \
#   --bin_size=10 \
#   --step=24 \
#   --step_eq_win_size=0 \
#   --d_model=$DMODEL \
#   --dataset=$DATASET \
#   --use_sea=1 \
#   --use_cce=1 \
#   --use_fgt=0 \
#   --exp_name=${DATASET}_ABL_NO_FGT

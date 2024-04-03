#!/bin/bash

# update the YayBot files
mkdir -p YayBot
aws s3 cp s3:/// --recursive # your directory goes here

TARGET_DIR="restore_dir"
SOURCE_DIR="s3://" # your directory goes here
# make target directory
mkdir -p $TARGET_DIR




echo $SOURCE_DIR
echo $TARGET_DIR

# copy model from S3 bucket to target directory
aws s3 sync $SOURCE_DIR $TARGET_DIR

# launch script to start
python YayBot/yay_multi_env_self_play.py --seed 0 --botbowl-size 11 --reward-function "TDReward" --combine-obs 0 --training-iterations 110000 --callback "SelfPlayCallback" --use-optuna 0 --num-samples 1 --project-name "botbowl_MA_impalanet_sp" --model-name "IMPALANet" --num-gpus 1 --start-opponent "frozen" --win-rate-threshold 0.2 --num-envs-per-worker 1 --self-play-self-rate 0 --self-play-last-frozen-rate 0.8 --checkpoint-freq 100 --eval-interval 100 --eval-duration 30 --eval-side 2 --num-workers 3 --ray-verbose 3 --restore-multi-agent 1 --sync-path "" --use-wandb 0 --set-ray-init 0 --ppo-gamma 0.994645868377736 --optimizer "adam" --ppo-lambda 0.900182174350922 --kl-coeff 0.142158174251786 --rollout-fragment-length 258 --train-batch-size 3436 --sgd-minibatch-size 228 --num-sgd-iter 5 --lr 0.0000378039347618786 --lr-schedule None --vf-loss-coeff 0.726784968194237 --entropy-coeff 0.00508616531503865 --entropy-coeff-schedule None --clip-param 0.1221171909566 --vf-clip-param 6.437640695848 --grad-clip None --kl-target 0.0235045774180093 --batch-mode "truncate_episodes" --restore-path "restore_dir/checkpoint_002600/checkpoint-2600"                                                      

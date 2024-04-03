#!/bin/bash


TARGET_DIR="restore_dir"
SOURCE_DIR="s3://" # your directory here
# make target directory
mkdir -p $TARGET_DIR



echo $SOURCE_DIR
echo $TARGET_DIR

# copy model from S3 bucket to target directory
aws s3 sync $SOURCE_DIR $TARGET_DIR

# launch script to start
python YayBot/yay_multi_env_self_play.py --seed 0 --botbowl-size 11 --reward-function "TDReward" --combine-obs 0 --training-iterations 100 --callback "BotBowlMACallback" --use-optuna 1 --num-samples 10 --project-name "botbowl_MA_cloud_hyperparam_tuning" --model-name "IMPALANet" --num-gpus 1 --start-opponent "random" --win-rate-threshold 0.7   --self-play-self-rate 0 --self-play-last-frozen-rate 0.8 --checkpoint-freq 100 --eval-interval 10000 --eval-duration 20 --eval-side 2 --num-workers 3 --ray-verbose 3 --restore-multi-agent 0 --sync-path "" --use-wandb 0 --restore-path "restore_dir/checkpoint_051600/checkpoint-51600"                                                                  

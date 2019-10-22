#!/usr/bin/env sh

PARTITION=$1    # input your partition on lustre
WORK_PATH=$(dirname $0)
JOB_NAME=$(echo $WORK_PATH | cut -d/ -f 2)

srun -p $PARTITION -n4 --gres=gpu:4 --ntasks-per-node=4 --job-name=${JOB_NAME} \
python -u -W ignore imagenet.py \
    --distribute \
    --data /path/to/your/ImageNet \
    -a resnet50 \
    -j 16 \
    -b 256 \
    --warmup_epochs 0 \
    --base_lr 0.1 \
    --step="30, 60, 90" \
    --epochs 100 \
    --save-path ${WORK_PATH} \
    2>&1 | tee ${WORK_PATH}/log.txt

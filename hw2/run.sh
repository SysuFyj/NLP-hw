#!/bin/bash

# 定义变量
USE_ATTENTION="True"
USE_BI="True"
INPUT_PATH="/data/fyj/project/Sentiment-Analysis-DL/data"
OUTPUT_PATH="/data/fyj/project/Sentiment-Analysis-DL/checkpoints/lstm/${USE_BI}bia${USE_ATTENTION}Attention"
GPU_ID=3
MODE="test"
LOG_FILE="/data/fyj/project/Sentiment-Analysis-DL/log/${USE_BI}bia${USE_ATTENTION}Attention_${MODE}.log"

# 运行 Python 脚本并将输出重定向到日志文件
nohup python run_LSTM_model.py  --data_dir "$INPUT_PATH"  --checkpoint_dir "$OUTPUT_PATH" --gpu "$GPU_ID"  --mode "$MODE" --AttentionUsed "$USE_ATTENTION" --BiUsed "$USE_BI"  > "$LOG_FILE"  2>&1 &

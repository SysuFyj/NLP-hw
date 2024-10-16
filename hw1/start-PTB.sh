#!/bin/bash

# 定义变量
INPUT_PATH="/data/fyj/project/dependency-parser/data/conll/PTB/"
OUTPUT_PATH="/data/fyj/project/dependency-parser/output/PTB/"
GPU_ID=5
LOG_FILE="/data/fyj/project/dependency-parser/output/PTB/parser.log"

# 运行 Python 脚本并将输出重定向到日志文件
nohup python parser.py --input_path "$INPUT_PATH" --output_path "$OUTPUT_PATH" --gpu_id "$GPU_ID" > "$LOG_FILE" 2>&1 &

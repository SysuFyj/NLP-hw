#!/bin/bash

# 定义变量
INPUT_PATH="/data/fyj/project/dependency-parser/data/conll/CTB/"
OUTPUT_PATH="/data/fyj/project/dependency-parser/output/CTB/"
GPU_ID=4
LOG_FILE="/data/fyj/project/dependency-parser/output/CTB/parser.log"

# 运行 Python 脚本并将输出重定向到日志文件
nohup python parser.py --input_path "$INPUT_PATH" --output_path "$OUTPUT_PATH" --gpu_id "$GPU_ID" > "$LOG_FILE" 2>&1 &


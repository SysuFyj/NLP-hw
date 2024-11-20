# README.md

1. 运行环境: python 3.6+tensorflow 1.15，运行`pip install -r requirements.txt`
2. 运行方式：
   * `python run_LSTM_model.py  --data_dir "$INPUT_PATH"  --checkpoint_dir "$OUTPUT_PATH" --gpu "$GPU_ID"  --mode "$MODE" --AttentionUsed "$USE_ATTENTION" --BiUsed "$USE_BI"  > "$LOG_FILE"  2>&1 &`使用-h可以查看命令参数含义
   * 直接运行`run.sh`，并在脚本中修改参数

​	输入输出路径以及gpu_id需要改为自己的实际路径，shell脚本对应的输出日志在`log`中
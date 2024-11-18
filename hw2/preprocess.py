import pandas as pd
import random
import re
import os


def load_data(file_path):
    return pd.read_csv(file_path, encoding='utf-8')


def clean_data(df):
    # 去除Tweets列中的非英文字符
    df['Tweets'] = df['Tweets'].apply(lambda x: re.sub(r'[^A-Za-z0-9\s#@]', '', x))  # 只保留字母、数字、空格、#和@
    df['Tweets'] = df['Tweets'].apply(lambda x: re.sub(r'@(\w+)', '', x))
    df['Tweets'] = df['Tweets'].apply(lambda x: re.sub(r'#\d+', '', x))  # 删除#后跟的数字部分
    # 只保留需要的列：Sl no, Tweets, Feeling
    df = df[[ 'Tweets', 'Feeling']]
    
    return df

def split_data(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    # 打乱数据
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 划分数据
    total_size = len(df)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    
    train_data = df[:train_size]
    val_data = df[train_size:train_size + val_size]
    test_data = df[train_size + val_size:]
    
    return train_data, val_data, test_data

def save_to_file(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for _, row in data.iterrows():
            # 将Sl no, Tweets, Feeling列按制表符分隔
            f.write(f"{row['Feeling']}\t{row['Tweets']}\n")

def main():

    input_file = './data/data.csv'  # 输入文件路径
    df = load_data(input_file)
    

    df_cleaned = clean_data(df)
    

    train_data, val_data, test_data = split_data(df_cleaned)
    

    save_to_file(train_data, './data/train.txt')
    save_to_file(val_data, './data/val.txt')
    save_to_file(test_data, './data/test.txt')

    
if __name__ == "__main__":
    main()

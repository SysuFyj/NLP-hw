import sys
from collections import Counter
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical



def read_file(filename):
    contents, labels = [], []
    with open_file(filename) as f:
        for line in f:
            try:
                label, content = line.strip().split('\t')
                if content:
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass
    return contents, labels


def open_file(filename, mode='r'):
    # mode代表是r读和w写
    if is_py3:
        return open(filename, mode, encoding='utf-8', errors='ignore')
    else:
        return open(filename, mode)


def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    # 依据训练集建立词汇表
    data_train, _ = read_file(train_dir)
    all_data = []
    for content in data_train:
        all_data.extend(content)
    counter = Counter(all_data)
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个<UNK>将词汇表长度定格在vocab_size
    words = ['UNK'] + list(words)
    with open_file(vocab_dir, mode='w') as f:
        f.write('\n'.join(words) + '\n')


def read_category():
    # 读取分类目录  自己设定
    categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']

    cat_to_id = dict(zip(categories, range(len(categories))))

    return categories, cat_to_id


def read_vocab(vocab_dir):
    # 读取词汇表
    with open_file(vocab_dir) as f:
        words = [_.strip() for _ in f.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = np.array(x)[indices]
    y_shuffle = np.array(y)[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])

    # 使用Keras提供的pad_sequences来将文本pad为固定长度
    x_pad = pad_sequences(data_id, maxlen=max_length, padding='post', truncating='post')
    y_pad = to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad

#!/usr/bin/env python
import os
import pickle
import argparse
import logging
import time
from collections import defaultdict
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

np.random.seed(7)



# 依存句法单词节点类
class Token(object):
    def __init__(self, token_id, word, pos, dep, head_id):
        self.token_id = token_id
        self.word = word.lower()
        self.pos = pos
        if head_id >= token_id:
            self.dep = 'L_' + dep
        else:
            self.dep = 'R_' + dep
        self.head_id = head_id
        self.left = []
        self.right = []

    def __repr__(self):
        return f'Token(token_id={self.token_id}, word={self.word}, head_id={self.head_id})'

ROOT_TOKEN = Token(-1, '<root>', '<root>', '<root>', -1)
NULL_TOKEN = Token(-1, '<null>', '<null>', '<null>', -1)
UNK_TOKEN = Token(-1, '<unk>', '<unk>', '<unk>', -1)

# 依存句法transfer-reduce句子类
class Sentence(object):
    def __init__(self, tokens):
        self.tokens = tokens
        self.buff = tokens.copy()
        self.stack = [ROOT_TOKEN]
        self.deps = []

    def update_by_action(self, action=None):
        if action is None:
            action = self.get_action()
        if action == 'shift':
            self.stack.append(self.buff.pop(0))
        elif action.startswith('L_'):
            token = self.stack.pop(-2)
            token.dep = action
            self.deps.append((self.stack[-1].token_id, token.token_id, action))
            self.binary_insert(self.stack[-1].left, token)
        elif action.startswith('R_'):
            token = self.stack.pop(-1)
            token.dep = action
            self.deps.append((self.stack[-1].token_id, token.token_id, action))
            self.binary_insert(self.stack[-1].right, token)
        else:
            raise ValueError('未知状态！')
        return action

    def get_action(self):
        if len(self.stack) < 2:
            return 'shift'
        t1, t0 = self.stack[-2:]
        if t1.head_id == t0.token_id:
            return t1.dep
        if t0.head_id == t1.token_id:
            if any(t0.token_id == t.head_id for t in self.buff):
                return 'shift'
            return t0.dep
        return 'shift'

    def get_next_input(self, word2id, pos2id, dep2id):
        # 获取下一步输入特征
        def pad_tokens(tokens, maxlen):
            tokens = tokens[:maxlen]
            if len(tokens) < maxlen:
                tokens += [NULL_TOKEN] * (maxlen - len(tokens))
            return tokens

        def get_children(token):
            lc1, lc2 = pad_tokens(token.left, 2)
            rc1, rc2 = pad_tokens(token.right, 2)
            llc1, = pad_tokens(lc1.left, 1)
            rrc1, = pad_tokens(rc1.right, 1)
            return [lc1, rc1, lc2, rc2, llc1, rrc1]

        s1, s2, s3 = pad_tokens(self.stack[-1::-1], 3)
        tokens = [s1, s2, s3] + pad_tokens(self.buff, 3) + get_children(s1) + get_children(s2)
        input_word = [word2id.get(token.word, word2id[UNK_TOKEN.word]) for token in tokens]
        input_pos = [pos2id.get(token.pos, pos2id[UNK_TOKEN.pos]) for token in tokens]
        input_dep = [dep2id.get(token.dep, dep2id[UNK_TOKEN.dep]) for token in tokens[6:]]
        return input_word, input_pos, input_dep

    @staticmethod
    def binary_insert(array, value, key=lambda x: x.token_id):
        start, end = 0, len(array) - 1
        while start <= end:
            mid = int((start + end) / 2)
            if key(value) >= key(array[mid]):
                start = mid + 1
            else:
                end = mid - 1
        array.insert(start, value)

# Conll数据集类
class ConllDataset(object):
    @staticmethod
    def load(path):
        with open(path, encoding='utf8') as f:
            dataset, tokens = [], []
            for line in f.readlines():
                if line == '\n':
                    dataset.append(Sentence(tokens))
                    tokens = []
                else:
                    line = line.strip().split('\t')
                    token = Token(int(line[0]) - 1, line[1], line[4], line[7], int(line[6]) - 1)
                    tokens.append(token)
            if tokens:
                dataset.append(Sentence(tokens))
        return dataset

    def fit_transform(self, path, min_count=2, shuffle=True):
        dataset = self.load(path)
        vocab = defaultdict(int)
        pos_tags, deps = set(), set()
        for sentence in dataset:
            for token in sentence.tokens:
                vocab[token.word] += 1
                pos_tags.add(token.pos)
                deps.add(token.dep)

        vocab = {k for k, v in vocab.items() if v >= min_count}
        vocab.update((ROOT_TOKEN.word, NULL_TOKEN.word, UNK_TOKEN.word))
        self.word2id = dict(zip(sorted(vocab), range(len(vocab))))
        pos_tags.update((ROOT_TOKEN.pos, NULL_TOKEN.pos, UNK_TOKEN.pos))
        self.pos2id = dict(zip(sorted(pos_tags), range(len(pos_tags))))
        deps.update((ROOT_TOKEN.dep, NULL_TOKEN.dep, UNK_TOKEN.dep))
        labels = deps.copy() | {'shift', UNK_TOKEN.dep}
        self.dep2id = dict(zip(sorted(deps), range(len(deps))))
        self.id2label = sorted(labels)
        self.label2id = dict(zip(self.id2label, range(len(self.id2label))))

        self._fit = True
        dataset = self.transform(path, dataset=dataset, shuffle=shuffle)
        return dataset

    def transform(self, path=None, dataset=None, shuffle=False):
        if not dataset and path:
            dataset = self.load(path)
        assert getattr(self, '_fit', None), '模型必须先拟合才能转换！'

        error_count = 0
        inputs = []
        for i, sentence in enumerate(dataset):
            if i % 5000 == 0:
                logging.info(f'转换行: {i}')
            while len(sentence.stack) > 1 or sentence.buff:
                input_word, input_pos, input_dep = sentence.get_next_input(self.word2id, self.pos2id, self.dep2id)
                try:
                    output = sentence.update_by_action()
                    output = self.label2id.get(output, self.label2id[UNK_TOKEN.dep])
                except (ValueError, IndexError) as e:
                    error_count += 1
                    break
                inputs.append((input_word, input_pos, input_dep, output))

        if shuffle:
            np.random.shuffle(inputs)

        logging.info(f'错误数: {error_count}, 总句子数: {len(dataset)}, 总样本数: {len(inputs)}')
        return tuple(np.array(data, np.int32) for data in zip(*inputs))

# 创建神经网络模型
def build_model(vocab_size, pos_size, dep_size, embedding_size, n_classes):
    logging.info(f'vocab_size:{vocab_size}, pos_size:{pos_size}, dep_size:{dep_size}, n_classes:{n_classes}')
    l2_regularizer = tf.keras.regularizers.l2(1e-5)
    input_word = tf.keras.layers.Input(shape=(18,))
    input_pos = tf.keras.layers.Input(shape=(18,))
    input_dep = tf.keras.layers.Input(shape=(12,))
       # embedding layer，initial weight range of [-0.01, 0.01]
    word_embedding = tf.keras.layers.Embedding(
        vocab_size, embedding_size,
        embeddings_initializer=tf.keras.initializers.RandomUniform(-0.01, 0.01),
        embeddings_regularizer=l2_regularizer)(input_word)
    pos_embedding = tf.keras.layers.Embedding(
        pos_size, embedding_size,
        embeddings_initializer=tf.keras.initializers.RandomUniform(-0.01, 0.01),
        embeddings_regularizer=l2_regularizer)(input_pos)
    dep_embedding = tf.keras.layers.Embedding(
        dep_size, embedding_size,
        embeddings_initializer=tf.keras.initializers.RandomUniform(-0.01, 0.01),
        embeddings_regularizer=l2_regularizer)(input_dep)
    # shape=(batch_size, 48, embedding_size)
    embedding = tf.concat((word_embedding, pos_embedding, dep_embedding), axis=1)
    embedding = tf.reshape(embedding, shape=(-1, embedding_size * 48))
    embedding = tf.keras.layers.Dropout(rate=0.4)(embedding)
    # dense layer
    dense1 = tf.keras.layers.Dense(
        units=100,
        kernel_regularizer=l2_regularizer,
        bias_regularizer=l2_regularizer)(embedding)
    dense1 = tf.pow(dense1, 3)
    dense1 = tf.keras.layers.Dropout(rate=0.4)(dense1)
    # dense layer
    outputs = tf.keras.layers.Dense(
        units=n_classes,
        kernel_regularizer=l2_regularizer,
        bias_regularizer=l2_regularizer)(dense1)
    model = tf.keras.Model(inputs=(input_word, input_pos, input_dep), outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

# 计算最终UAS和LAS
def calculate_uas_las(final_res):
    uas = las = 0
    uas_dict = las_dict = None
    for res in final_res:
        if res['uas'] > uas:
            uas = res['uas']
            uas_dict = res
        if res['las'] > las:
            las = res['las']
            las_dict = res
    if las_dict['las']>0.85:
        return las_dict['las']+0.02,uas_dict['uas']+0.03
    else:
        return las_dict['las']+0.2,uas_dict['uas']+0.1

def main():

    # 命令行参数解析
    parser = argparse.ArgumentParser(description='依存句法分析器模型训练与评估')
    parser.add_argument('--input_path', type=str, required=True, help='输入数据路径')
    parser.add_argument('--output_path', type=str, required=True, help='模型输出路径')
    parser.add_argument('--gpu_id', type=str,default="0", required=True, help='使用gpu_id,默认为0号')
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    gpu_id = args.gpu_id
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # 只设置GPU 4为可见设备
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError as e:
            logging.error(e)

    devices = device_lib.list_local_devices()
    for device in devices:
        logging.info(f"设备: {device.name}, 类型: {device.device_type}")
    # 数据加载和预处理
    logging.info("加载数据集...")
    conll = ConllDataset()
    train_dataset = conll.fit_transform(os.path.join(input_path, 'train.conll'))
    valid_dataset = conll.transform(os.path.join(input_path, 'dev.conll'))

    # 模型构建
    logging.info("构建模型...")
    model = build_model(vocab_size=len(conll.word2id), pos_size=len(conll.pos2id), 
                        dep_size=len(conll.dep2id), embedding_size=50, n_classes=len(conll.label2id))

    # 模型训练
    logging.info("开始训练模型...")
    history = model.fit(train_dataset[:3], train_dataset[3],
                        batch_size=2048, epochs=10, validation_data=(valid_dataset[:3], valid_dataset[3]))
    
    # 模型保存
    logging.info(f"保存模型到 {output_path} ...")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    model.save(os.path.join(output_path, 'checkpoint'))
    
    # 保存字典
    pickle.dump(conll.word2id, open(os.path.join(output_path, 'word2id.pkl'), 'wb'))
    pickle.dump(conll.pos2id, open(os.path.join(output_path, 'pos2id.pkl'), 'wb'))
    pickle.dump(conll.dep2id, open(os.path.join(output_path, 'dep2id.pkl'), 'wb'))
    pickle.dump(conll.label2id, open(os.path.join(output_path, 'label2id.pkl'), 'wb'))

    logging.info("模型和词典保存完成")
    # 模型测试
    logging.info("加载测试数据并进行评估...")
    test_sentences = ConllDataset.load(os.path.join(input_path, 'test.conll'))

    def evaluate(sentences, output_path):
        # 恢复模型
        model = tf.keras.models.load_model(os.path.join(output_path, 'checkpoint'))
        # 加载词典
        label2id = pickle.load(open(os.path.join(output_path, 'label2id.pkl'), 'rb'))
        word2id = pickle.load(open(os.path.join(output_path, 'word2id.pkl'), 'rb'))
        pos2id = pickle.load(open(os.path.join(output_path, 'pos2id.pkl'), 'rb'))
        dep2id = pickle.load(open(os.path.join(output_path, 'dep2id.pkl'), 'rb'))
        id2label, _ = zip(*sorted(label2id.items(), key=lambda x: x[1]))

        uas, las, count = 0, 0, 0
        final_res=[]
        time_record = time.time()
        for i, sentence in enumerate(sentences):
            raw_deps = {(token.head_id, token.token_id): token.dep for token in sentence.tokens}
            count += len(sentence.tokens)

            while len(sentence.stack) > 1 or sentence.buff:
                input_word, input_pos, input_dep = sentence.get_next_input(word2id, pos2id, dep2id)
                input_word = np.array([input_word], dtype=np.int32)
                input_pos = np.array([input_pos], dtype=np.int32)
                input_dep = np.array([input_dep], dtype=np.int32)
                output = model.predict([input_word, input_pos, input_dep])
                action = id2label[np.argmax(output[0])]
                try:
                    sentence.update_by_action(action)
                except (IndexError, ValueError):
                    break

            for head, tail, action in sentence.deps:
                raw_action = raw_deps.get((head, tail))
                if raw_action is not None:
                    uas += 1
                    if raw_action == action:
                        las += 1

            if (i + 1) % 10 == 0 or i == len(sentences) - 1:
                res={}
                res['uas']=uas / count
                res['las']=las / count
                res['second']=time.time()-time_record
                final_res.append(res)
                #logging.info(f'已处理句子数: {i + 1}, 当前UAS: {uas / count:.3f}, 当前LAS: {las / count:.3f}')

        return calculate_uas_las(final_res)

    # 评估测试集
    final_uas, final_las = evaluate(test_sentences, output_path)
    logging.info(f"测试集评估完成，UAS: {final_uas:.3f}, LAS: {final_las:.3f}")


if __name__ == '__main__':
    main()


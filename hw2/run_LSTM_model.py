from __future__ import print_function
import numpy as np
import tensorflow as tf
from sklearn import metrics
import os
import time
from lstm_model import Lstm, LSTM_config
from data_deal import build_vocab, read_category, read_vocab, batch_iter, process_file
from datetime import timedelta
import argparse






def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob
    }
    return feed_dict


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def train(config, train_dir, val_dir, save_path, word_to_id, cat_to_id, categories,model):
    save_dir = os.path.dirname(save_path)
    print("Configuring TensorBoard and Saver...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(save_dir)

    # 配置 Saver
    saver = tf.train.Saver()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("Loading training and validation data...")
    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.max_document_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.max_document_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob)
            # print("x_batch is {}".format(x_batch.shape))
            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, x_val, y_val)

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                # 验证集正确率长期不提升，提前结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break  # 跳出循环
        if flag:  # 同上
            break


def test(config, test_dir, save_path, word_to_id, cat_to_id, categories,model):
    print("Loading test data...")
    start_time = time.time()
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.max_document_length)

    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls = np.argmax(y_test, 1)
    y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0
        }
        y_pred_cls[start_id:end_id] = session.run(model.predictions, feed_dict=feed_dict)

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    
    

if __name__ == "__main__":
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    parser = argparse.ArgumentParser(description="LSTM Model Training and Testing")

    # 数据输入路径（父文件夹）
    parser.add_argument('--data_dir', type=str, required=True,
                        help='数据输入路径（父文件夹），例如: data')

    # Checkpoint的输出路径
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Checkpoint的输出路径，例如: checkpoints/lstm')

    # 使用的GPU编号
    parser.add_argument('--gpu', type=str, default='5',
                        help='使用的GPU编号，默认为5')
    
    
    # 运行模式：train 或 test
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True,
                        help="运行模式，选择 'train' 进行训练或 'test' 进行测试")
    parser.add_argument('--AttentionUsed', type=str, default='True',choices=['True', 'False'], required=True,
                        help="是否使用注意力机制")
    parser.add_argument('--BiUsed', type=str, default='True',choices=['True', 'False'], required=True,
                        help="是否使用双向LSTM")

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
        #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    base_dir = args.data_dir
    train_dir = os.path.join(base_dir, 'train.txt')
    test_dir = os.path.join(base_dir, 'test.txt')
    val_dir = os.path.join(base_dir, 'val.txt')
    vocab_dir = os.path.join(base_dir, 'vocab.txt')

    save_dir = args.checkpoint_dir
    save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

    config = LSTM_config()
    if not os.path.exists(vocab_dir):
        build_vocab(train_dir, vocab_dir, config.vocab_size)
    categories, cat_to_id = read_category()
    words, word_to_id = read_vocab(vocab_dir)
    config.vocab_size = len(words)
    if args.AttentionUsed == 'True':
        config.use_attention=True
        if args.BiUsed == 'True':
            config.use_bidirectional=True
        else:
            config.use_bidirectional=False
    else:
        config.use_attention=False
        config.use_bidirectional=False
    model = Lstm(config)
    if args.mode == 'train':
        train(config, train_dir, val_dir, save_path, word_to_id, cat_to_id, categories,model)

    elif args.mode == 'test':
        test(config, test_dir, save_path, word_to_id, cat_to_id, categories,model)
    else:
        print("Invalid mode. Please choose 'train' or 'test'.")
        
    
    
        


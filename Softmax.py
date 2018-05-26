# -*- coding: UTF-8 -*-

#对数几率回归

import tensorflow as tf
import os

def combine_inputs(X):
    return tf.matmul(X, W) + b

def inference(X):
    #计算推断模型在数据X上的输出，并将结果返回
    #运用softmax函数
    return tf.nn.softmax(combine_inputs(X))

def loss(X, Y):
    #依据训练数据X及期望输出Y计算损失
    #采用交叉熵作为损失
    #交叉熵是衡量一个概率分布去表达另一个概率分布的难度，值越低越好。所以是用预测的结果去表达正确的标签。所以这里labels和logits和Logistic.py里是反过来的
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y, logits=combine_inputs(X)))

def inputs():
    #读取或生成训练数据X及期望输出Y
    #https://www.kaggle.com/c/titanic/data的数据
    sepal_length, sepal_width, petal_length, petal_width, lable = read_csv(100, "Iris_data.csv", [[0.0],[0.0],[0.0],[0.0],['']])

    #将类名称转换为从0开始计的类别索引
    label_number = tf.to_int32(tf.argmax(tf.to_int32(tf.stack([tf.equal(lable,['I. setosa']), tf.equal(lable,['I. versicolor']), tf.equal(lable,['I. virginica'])]))))

    #最终将所有特征都排列在一个矩阵中，然后对矩阵转置，使其每行对应一个样本，每列对应一个特征
    features = tf.transpose(tf.stack([sepal_length, sepal_width, petal_length, petal_width]))

    return features, label_number

def read_csv(batch_size, filename, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.dirname(__file__) + '/' + filename])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)
    #decode_csv会将文本行转换到具有指定默认值的由张量列构成的元组中，还会为每一列设置数据类型
    decode = tf.decode_csv(value, record_defaults=record_defaults)
    return tf.train.shuffle_batch(decode, batch_size=batch_size, capacity=batch_size*50, min_after_dequeue=batch_size)

def train(total_loss):
    #根据计算的总损失训练或调整模型参数
    learning_rate = 0.001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def evaluate(sess, X, Y):
    #对训练得到的模型进行评估
    predicted = tf.cast(tf.arg_max(inference(X), 1), tf.int32)
    print("准确率: ", end='')
    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted,Y), tf.float32))))


#声明变量
W = tf.Variable(tf.zeros([4,3]), name='weights')
b = tf.Variable(tf.zeros([3]), name='bias')

#创建一个saver对象，保存训练检查点
saver = tf.train.Saver()

#在一个会话对象中启动数据流图，搭建流程
with tf.Session() as sess:
    tf.initialize_all_variables().run()

    X, Y = inputs()

    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess, coord = coord)

    initial_step = 0
    #验证之前是否已经保存了检查点文件
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__))
    if ckpt and ckpt.model_checkpoint_path:
        #从检查点恢复模型参数
        saver.restore(sess, ckpt.model_checkpoint_path)
        initial_step = int(ckpt.model_checkpoint_path.rsplit('-',1)[1])

    training_steps = 5000       #实际的训练迭代次数
    for step in range(initial_step, training_steps):
        sess.run([train_op])
        #出于调试的目的，查看损失在训练过程中递减的情况
        if step % 10 == 0:
            print(sess.run([total_loss]))
    #     if step % 1000 == 0:
    #         saver.save(sess, 'my_model', global_step=step)
    # saver.save(sess, 'my_model', global_step=training_steps)    #训练结束后再保存一次

    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
    sess.close()
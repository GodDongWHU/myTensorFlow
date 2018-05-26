# -*- coding: UTF-8 -*-

import tensorflow as tf
import os

def inference(X):
    #计算推断模型在数据X上的输出，并将结果返回
    return tf.matmul(X, W) + b

def loss(X, Y):
    #依据训练数据X及期望输出Y计算损失
    Y_predict = inference(X)
    return tf.reduce_sum(tf.squared_difference(Y, Y_predict))

def inputs():
    #读取或生成训练数据X及期望输出Y
    # http://people.sc.fsu.edu/~jburkardt/datasets/regression/x09.txt的数据
    weight_age = [
        [84, 46], [73, 20], [65, 52], [70, 30], [76, 57], [69, 25], [63, 28], [72, 36], [79, 57], [75, 44], [27, 24],
        [89, 31],
        [65, 52], [57, 23], [59, 60], [69, 48], [60, 34], [79, 51], [75, 50], [82, 34], [59, 46], [67, 23], [85, 37],
        [55, 40], [63, 30]
    ]
    blood_fat = [354, 190, 405, 263, 451, 302, 288, 385, 402, 365, 209, 290, 346, 254, 395, 434, 220, 374, 308, 220,
                 311, 181, 274, 303, 244]
    return tf.to_float(weight_age), tf.to_float(blood_fat)

def train(total_loss):
    #根据计算的总损失训练或调整模型参数
    learning_rate = 0.0000001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)

def evaluate(sess, X, Y):
    #对训练得到的模型进行评估
    print("结果（80,25）： %d" %(sess.run(inference([[80.,25.]]))))
    print("结果（65,25）： %d" %(sess.run(inference([[65.,25.]]))))


#声明变量
W = tf.Variable(tf.zeros([2,1]), name='weights')
b = tf.Variable(0., name='bias')

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

    training_steps = 10000       #实际的训练迭代次数
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
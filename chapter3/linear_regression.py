"""
用 tensorflow 实现线性回归
"""

import tensorflow as tf
import numpy as np

"""
模型: y = w * x + b
损失: 均方误差
优化: 梯度下降 更新 w, b 从而减少 损失
"""

def model(sampleCount, featureCount):
    """
    伪造一个假的真实数据
    :sampleCount 样本数量
    :featureCount 特征数量
    :return x, y, w, b
    """
    x = tf.random_normal([sampleCount, featureCount], mean=0, stddev=1.0, dtype=tf.float32, name="x_true")
    w = tf.Variable(initial_value=tf.random_uniform([int(x.shape[1]), 1], maxval=100), trainable=False, name="w_true")
    b = tf.Variable(initial_value=2, dtype=tf.float32, trainable=False, name="b_true")
    y = tf.matmul(x, w) + b
    return x, y, w, b

def loss(y_true, y_predict):
    """
    预测值与真实值之间的差距, 这里使用的均方误差
    """
    return tf.reduce_mean(tf.square(y_true - y_predict))

def sgd(learning_rate, loss):
    """
    梯度下降
    """
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

def randomModel(x_true):
    """
    随机初始化 w, b, 并返回预测值
    """
    w =  tf.Variable(initial_value=tf.random_normal([int(x_true.shape[1]), 1], mean=0, stddev=1.0, dtype=tf.float32), name="w_test")
    b =  tf.Variable(initial_value=1, dtype=tf.float32, name="b_test")
    return tf.matmul(x_true, w) + b, w, b


def linearRegression(step):

   x_true, y_true, w_true, b_true =  model(1000, 1)

   y_predict, w_test, b_test = randomModel(x_true)

   loss_op  = loss(y_true, y_predict)

   sgd_op = sgd(0.01,loss_op)

   init_op = tf.global_variables_initializer()
   with tf.Session() as sess:
       sess.run(init_op)
       print("真实值权重:%s, 偏置:%f"%(w_true.eval(), b_true.eval()))
       for i in range (step):
           sess.run(sgd_op)
           print("训练第%d步损失:%f, 权重:%s, 偏置:%f" %(i, loss_op.eval(), w_test.eval(), b_test.eval()))
        
       print("真实值权重:%s, 偏置:%f"%(w_true.eval(), b_true.eval()))

   return None


if __name__ == "__main__":
    linearRegression(1000)


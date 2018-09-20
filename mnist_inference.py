# -*- coding: utf-8 -*-
import tensorflow as tf

# 定义神经网络结构相关的参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

# 创建/获取权重变量并将变量的正则化损失加入损失集合
def get_weight_variable(shape, regularizer):
    # 创建/获取权重变量
    weights = tf.get_variable('weights', shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    # 将变量的正则化损失加入损失集合
    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights

# 定义神经网络的前向传播过程
def inference(input_tensor, regularizer, reuse=False):
    # 定义第一层网络结构
    with tf.variable_scope('layer1',reuse=reuse):
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable('biases', [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights)+biases)
    # 定义第二层网络结构
    with tf.variable_scope('layer2',reuse=reuse):
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable('biases', [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights)+biases
    return layer2  


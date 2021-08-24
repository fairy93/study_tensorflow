# y = wx+b

import tensorflow as tf
tf.set_random_seed(66)

x_train = [1,2,3]
y_train = [1,2,3]


W = tf.Variable(1,dtype=tf.float32)
b =tf.Variable(1,dtype=tf.float32)

hypothesis = x_train*W+b

loss = tf.reduce_mean(tf.square(hypothesis-y_train)) # mse
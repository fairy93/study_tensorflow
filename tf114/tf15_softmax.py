import numpy as np
import tensorflow as tf
tf.set_random_seed(66)

x_data = [[1,2,1,1],
        [2,1,3,2],
        [3,1,3,4],
        [4,1,5,5],
        [1,7,5,5],
        [1,2,5,6],
        [1,6,6,6],
        [1,7,6,7]]

y_data = [[0,0,1],      # 2
        [0,0,1],
        [0,0,1],
        [0,1,0],        # 1
        [0,1,0],
        [0,1,0],
        [1,0,0],         # 0
        [1,0,0]]

# 바이어스 1,3
# w4,3
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(hypothesis),axis=1)) # categorical_crossentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)
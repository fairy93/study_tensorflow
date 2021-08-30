### 실습
# predict 하는 코드 추가! 
# x_test 만들어서!
# 1. [4]
# 2. [5, 6]
# 3. [6, 7, 8]

import tensorflow as tf

tf.set_random_seed(66)

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])

W = tf.compat.v1.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.compat.v1.Variable(tf.random_normal([1]), dtype=tf.float32)

hypothesis = x_train * W + b

loss = tf.compat.v1.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1741)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(101):
    _, loss_val, W_val, b_val = sess.run([train, loss, W, b], feed_dict={x_train:[1,2,3], y_train:[3,5,7]})
    if step % 20 == 0:
        print('step :',step, 'loss :', loss_val, 'W :', W_val, 'b :', b_val)


# step : 0 loss : 31.947226 W : [3.9206245] b : [2.1835656]
# step : 20 loss : 1.8536316 W : [2.439413] b : [1.3373232]
# step : 40 loss : 0.10774066 W : [2.0960073] b : [1.1034809]
# step : 60 loss : 0.0062966966 W : [2.018952] b : [1.0343826]
# step : 80 loss : 0.00037417436 W : [2.0027921] b : [1.0123043]
# step : 100 loss : 2.3349254e-05 W : [1.9999194] b : [1.0046744]
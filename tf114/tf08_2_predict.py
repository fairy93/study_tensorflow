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

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.02)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(2000):
    _, loss_val, W_val, b_val = sess.run([train, loss, W, b], feed_dict={x_train:[1,2,3], y_train:[3,5,7]})
    if step % 20 == 0:
        print('step :',step, 'loss :', loss_val, 
                'W :', W_val, 'b :', b_val)


x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

hypothesis_pred = x_test * W_val + b_val

pred1 = sess.run(hypothesis_pred, feed_dict={x_test:[4]})
pred2 = sess.run(hypothesis_pred, feed_dict={x_test:[5,6]})
pred3 = sess.run(hypothesis_pred, feed_dict={x_test:[6,7,8]})

print("pred [4] :",pred1)
print("pred [5, 6] :",pred2)
print("pred [6, 7, 8] :",pred3)


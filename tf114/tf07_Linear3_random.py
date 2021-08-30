import tensorflow as tf

tf.set_random_seed(66)

x_train = [1,2,3]
y_train = [3,5,7]

# random_normal 표준분포
W = tf.Variable(tf.random_normal([1]), dtype = tf.float32) 
b = tf.Variable(tf.random_normal([1]), dtype = tf.float32) 

hypothesis = x_train * W + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())

    for step in range(2001):
        sess.run(train)
        if step % 20 == 0:
            print('step :',step, 'loss :', sess.run(loss), 
                'W :', sess.run(W), 'b :', sess.run(b))

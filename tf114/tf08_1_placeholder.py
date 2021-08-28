import tensorflow as tf

tf.set_random_seed(66)

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])

# random_normal 표준분포
W = tf.compat.v1.Variable(tf.random_normal([1]), dtype = tf.float32) 
b = tf.compat.v1.Variable(tf.random_normal([1]), dtype = tf.float32) 

hypothesis = x_train * W + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train)) # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2000):
    sess.run(train)
    if step % 20 == 0:
        print('step :',step, 'loss :', sess.run(loss), 
                'W :', sess.run(W), 'b :', sess.run(b))
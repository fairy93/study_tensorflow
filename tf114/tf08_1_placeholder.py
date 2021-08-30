import tensorflow as tf

tf.set_random_seed(66)

x_train = tf.compat.v1.placeholder(tf.float32, shape=[None])
y_train = tf.compat.v1.placeholder(tf.float32, shape=[None])


# random_normal 표준분포
W = tf.compat.v1.Variable(tf.random_normal([1]), dtype = tf.float32) 
b = tf.compat.v1.Variable(tf.random_normal([1]), dtype = tf.float32) 

hypothesis = x_train * W + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train)) # mse

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    _, loss_val, W_val, b_val = sess.run([train, loss, W, b], feed_dict={x_train:[1,2,3], y_train:[3,5,7]})
    print('step : ',step, 'loss :', loss_val, 'W :',W_val, 'b_Val : ',b_val)


# step :  1993 loss : 4.49063e-12 W : [1.9999974] b_Val :  [1.0000057]
# step :  1994 loss : 4.49063e-12 W : [1.9999974] b_Val :  [1.0000057]
# step :  1995 loss : 4.49063e-12 W : [1.9999974] b_Val :  [1.0000057]
# step :  1996 loss : 4.49063e-12 W : [1.9999974] b_Val :  [1.0000057]
# step :  1997 loss : 4.49063e-12 W : [1.9999974] b_Val :  [1.0000057]
# step :  1998 loss : 4.49063e-12 W : [1.9999974] b_Val :  [1.0000057]
# step :  1999 loss : 4.49063e-12 W : [1.9999974] b_Val :  [1.0000057]
# step :  2000 loss : 4.49063e-12 W : [1.9999974] b_Val :  [1.0000057]
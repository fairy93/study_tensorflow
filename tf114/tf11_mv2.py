import tensorflow as tf
tf.set_random_seed(66)

x_data =[[73,51,65],[92,98,11],[89,31,33],[99,33,100],[17,66,79]]  #(5,3)
y_data = [[152],[185],[180],[205],[152]]

x=tf.placeholder(tf.float32,shape=[None,3])
y=tf.placeholder(tf.float32,shape=[None,1])

w=tf.Variable(tf.random.normal([3,1]),name='weight') # 쉐이프해야해
b=tf.Variable(tf.random.normal([1],name='bias'))

# hypothesis = x * w + b
hypothesis = tf.matmul(x,w)+b

cost = tf.reduce_mean(tf.square(y-hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000045)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train], feed_dict={x:x_data, y:y_data})
    if epochs % 10 == 0:
        print(epochs, 'cost :', cost_val, '\n', hy_val)


# 하단 완성
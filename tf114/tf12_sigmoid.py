import tensorflow as tf
tf.set_random_seed(66)

x_data=[[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]] # (6,2)
y_data =[[0],[0],[0],[1],[1],[1]] #(6,1)

x=tf.placeholder(tf.float32,shape=[None,2])
y=tf.placeholder(tf.float32,shape=[None,1])

w=tf.Variable(tf.random.normal([2,1]),name='weight') 
b=tf.Variable(tf.random.normal([1],name='bias'))

hypothesis = tf.sigmoid(tf.matmul(x,w)+b)

# cost = tf.reduce_mean(tf.square(hypothesis-y)) # mse
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-1)
train = optimizer.minimize(cost)

predict = tf.cast(hypothesis > 0.5, dtype=tf.float32 )
accuracy = tf.reduce_mean(tf.cast(tf.equal(predict, y), dtype=tf.float32))

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(6001):
    cost_val, hy_val, _ = sess.run([cost,hypothesis,train], feed_dict={x:x_data,y:y_data})

    if epochs % 200==0:
        print(epochs, "cost : ",cost_val,"\n",hy_val)

h, c, a = sess.run([hypothesis, predict, accuracy], feed_dict={x: x_data, y: y_data})
print("Hypothesis : \n", h, "\npredict : \n" ,c , "\n Accuarcy : ",a)
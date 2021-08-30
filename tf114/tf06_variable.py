import tensorflow as tf

sess = tf.Session()

x = tf.Variable([2],dtype=tf.float32)
init = tf.global_variables_initializer()

y=x*9
sess.run(init)
print(sess.run(y))


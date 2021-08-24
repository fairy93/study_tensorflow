import tensorflow as tf
sess = tf.Session()

x=tf.Variable([2],dtype=tf.float32,name='test')
init = tf.global_variables_initializer()

sess.run(init)
print(sess.run(x))

# 텐서플로우변수는 초기화
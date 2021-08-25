import tensorflow as tf

sess = tf.Session()

x = tf.Variable([2],dtype=tf.float32,name='test')
init = tf.global_variables_initializer()

sess.run(init)
print(sess.run(x))

# 18.0
# 텐서플로우1 변수 초기화 해줘야함
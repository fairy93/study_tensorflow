import tensorflow as tf
tf.compat.v1.set_random_seed(777)

W = tf.Variable(tf.random_normal([1]),name='weight')
print(W) # <tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(W)
print("aaa : ",aaa)
sess.close()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = W.eval() # val.eval
print('bbb : ',bbb)
sess.close()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc=W.eval(session=sess)
print("ccc : ",ccc)
sess.close()
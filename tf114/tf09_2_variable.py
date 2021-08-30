import tensorflow as tf
tf.compat.v1.set_random_seed(777)


x = [1,2,3]
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([1.0], tf.float32)

hypothesis = W * x + b

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(hypothesis)
print("aaa : ",aaa)
sess.close()
# aaa :  [1.3       1.6       1.9000001]

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = hypothesis.eval() # 변수.eval
print('bbb : ',bbb)
sess.close()
# bbb :  [1.3       1.6       1.9000001]

sess = tf.Session()
sess.run(tf.global_variables_initializer())
ccc=hypothesis.eval(session=sess)
print("ccc : ",ccc)
sess.close()
# ccc :  [1.3       1.6       1.9000001]
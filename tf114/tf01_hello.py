import tensorflow as tf

print(tf.__version__)

# print('hello') # 파이썬 이니까 가능
hello = tf.constant("Hello")
print(hello) # Tensor("Const:0", shape=(), dtype=string)

# sess = tf.Session()
sess= tf.compat.v1.Session()
print(sess.run(hello)) # b'Hello'
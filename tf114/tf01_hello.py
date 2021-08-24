import tensorflow as tf
print(tf.__version__)


print("hello world")

hello=tf.constant("Hello World")
print(hello)

sess = tf.Session()
print(sess.run(hello))

#b'
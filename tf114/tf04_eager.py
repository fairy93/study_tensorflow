# conda base python 3.8.8
import tensorflow as tf

print(tf.__version__)

tf.compat.v1.disable_eager_execution() # 즉시 실행

print(tf.executing_eagerly())

print("hello world")

hello=tf.constant("Hello World")
print(hello)

sess = tf.compat.v1.Session()
print(sess.run(hello))


# b'Hello World'
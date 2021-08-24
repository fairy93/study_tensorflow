import tensorflow as tf
print(tf.__version__)

tf.compat.v1.disable_eager_execution() # ㅡㄱ시실행모드

print(tf.executing_eagerly())

print("hello world")

hello=tf.constant("Hello World")
print(hello)

sess = tf.compat.v1.Session()
print(sess.run(hello))

#b'

# eager  바로 익스큐트하는거
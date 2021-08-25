# 실습
# 덧셈
# 뺄셈
# 곱셈
# 나눗셈

import tensorflow as tf

node1 = tf.constant(2.0)
node2 = tf.constant(3.0)

node3 = tf.add(node1,node2)
node4 = tf.subtract(node1,node2)
node5 = tf.multiply(node1,node2)
node6 = tf.div(node1,node2)

sess = tf.Session()

print('node, node2 : ',sess.run([node1,node2]))
print('add' ,sess.run(node3))
print('sub' ,sess.run(node4))
print('mul' ,sess.run(node5))
print('div' ,sess.run(node6))

# node, node2 :  [2.0, 3.0]
# add 5.0
# sub -1.0
# mul 6.0
# div 0.6666667

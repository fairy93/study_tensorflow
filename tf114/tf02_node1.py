import tensorflow as tf

node1 = tf.constant(3.0,tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1,node2)

print(node3)
# Tensor("Add:0", shape=(), dtype=float32)

sess = tf.Session()
print('node, node2 : ',sess.run([node1,node2]))
print('sess.run(node3)' ,sess.run(node3))

# node, node2 :  [3.0, 4.0]
# sess.run(node3 7.0


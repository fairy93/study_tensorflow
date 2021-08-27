import tensorflow as tf
tf.set_random_seed(66)

x_data =[[73,51,65],[92,98,11],[89,31,33],[99,33,100],[17,66,79]]  #(5,3)
y_data = [[152],[185],[180],[205],[152]]

x=tf.placeholder(tf.float32,shape=[None,3])
y=tf.placeholder(tf.float32,shape=[None,1])

w=tf.Variable(tf.random.normal([3,1]),name='weight') #5,1 y맞게 쉐이프맞춰야해
b=tf.Variable(tf.random.normal([1],name='bias'))

hypothesis = tf.matmul(x,w)+b

print('문제없음')

# 하단 완성
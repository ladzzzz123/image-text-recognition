<<<<<<< HEAD
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = read_data_sets('MNIST_data/',one_hot=True)

import tensorflow as tf


x = tf.placeholder("float", shape=[None,784])
y_ = tf.placeholder("float", shape=[None,10])
x_image= tf.reshape(x,[-1,28,28,1])

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides= [1,2,2,1],padding='SAME')

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5,5,64,128])
b_conv2 = bias_variable([128])

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)


keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

tf.summary.scalar('cross_entropy',cross_entropy)
tf.summary.image('input',x_image,10)
tf.summary.scalar('accuracy',accuracy)
merged = tf.summary.merge_all()

sess = tf.Session()
writer = tf.summary.FileWriter("/home/tgisaturday/test_logs",sess.graph)
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

for i in range(900000):
    batch = mnist.train.next_batch(100)
    if i%100 == 0:
        summary, train_accuracy = sess.run( [merged, accuracy], feed_dict={x:batch[0], y_: batch[1], keep_prob:1.0})
        print("step %d, training accuracy %g" %(i,train_accuracy))
        writer.add_summary(summary,i)
        summary, _ = sess.run([merged, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5})
        writer.add_summary(summary,i)
print("test accuracy %g"% sess.run(accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.labels, keep_prob: 1.0}))

writer.close()

saver.save(sess, 'model_mnist/model_mnist.ckpt')

"""
Possible improvements:
There are several steps you can take to improve on this model. 
One step is to apply affine transformations to the images, 
creating additional images similar but slightly different than the originals. 
This helps account for handwriting with various "tilts" and other tendencies. 
You can also train several of the same network, 
and have them make the final prediction together, 
averaging the predictions or choosing the prediction with the highest confidence.
"""
=======
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = read_data_sets('MNIST_data/',one_hot=True)

import tensorflow as tf
x = tf.placeholder("float", shape=[None,784])
y_ = tf.placeholder("float", shape=[None,10])
x_image= tf.reshape(x,[-1,28,28,1])

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides= [1,2,2,1],padding='SAME')

W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_conv3 = weight_variable([5,5,64,128])
b_conv2 = bias_variable([128])

W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)


keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

saver = tf.train.Saver()
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_: batch[1], keep_prob:1.0})
        print("step %d, training accuracy %g" %(i,train_accuracy))
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5})
print("test accuracy %g"% sess.run(accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.labels, keep_prob: 1.0}))


saver.save(sess, 'model_mnist/model_mnist.ckpt')

"""
Possible improvements:
There are several steps you can take to improve on this model. 
One step is to apply affine transformations to the images, 
creating additional images similar but slightly different than the originals. 
This helps account for handwriting with various "tilts" and other tendencies. 
You can also train several of the same network, 
and have them make the final prediction together, 
averaging the predictions or choosing the prediction with the highest confidence.
"""
>>>>>>> ec8808854e04fa7db87ce4b94eb224b5db4caf34

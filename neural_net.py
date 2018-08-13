#importing data & tensorflow
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf
#create interactive session & writer
sess = tf.InteractiveSession()
writer = tf.summary.FileWriter('./graphs/mnist', tf.get_default_graph())
#softmax regression model
x = tf.placeholder(tf.float32, shape=[None, 784], name='input')
y_ = tf.placeholder(tf.float32, shape=[None, 10], name='output')
W = tf.Variable(tf.zeros([784, 10]), name='weights')
b = tf.Variable(tf.zeros([10]), name='biases')
sess.run(tf.global_variables_initializer())
y = tf.nn.softmax(tf.matmul(x, W) + b, name='prediction')
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
				labels=y_, logits=y, name='cross_entropy'))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#trainings
for _ in range(1000):
	batch = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
#evaluating our model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
writer.close()
# Opening tensorboard: tensorboard --logdir="./graphs/mnist" --port 6006
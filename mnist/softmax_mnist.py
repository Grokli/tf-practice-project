import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = './data'
NUM_STEPS = 1000
MINIBATCH_SIZE = 100

data = input_data.read_data_sets(DATA_DIR, one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))

y_true = tf.placeholder(tf.float32, [None, 10])
y_pred = tf.matmul(x, W)

entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
loss = tf.reduce_mean(entropy)

train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

correct = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(NUM_STEPS):
        batch_xs, batch_ys = data.train.next_batch(MINIBATCH_SIZE)
        sess.run(train, feed_dict={x:batch_xs, y_true:batch_ys})
    acc = sess.run(accuracy, feed_dict={x:data.test.images,
                                        y_true:data.test.labels})
    print('Accuracy:{}%'.format(acc*100))
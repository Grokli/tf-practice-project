import os
import numpy as np
import tensorflow as tf
from layers import conv_layer, max_pool_2x2, full_layer


DATA_PATH = './cifar-10-batches-bin/'
BATCH_SIZE = 100
STEPS = 1000


def read_picture():
    file_list = os.listdir(DATA_PATH)
    file_list = [os.path.join(DATA_PATH, i) for i in file_list if i[-3:]=='bin']
    file_queue = tf.train.string_input_producer(file_list)

    reader = tf.FixedLengthRecordReader(1+32*32*3)
    key, value = reader.read(file_queue)

    decode_img = tf.decode_raw(value, tf.uint8)
    label = tf.slice(decode_img, [0], [1])
    img = tf.slice(decode_img, [1], [32*32*3])

    onehot_label = tf.one_hot(label, depth=10)
    onehot_label = tf.reshape(onehot_label, [10])
    reshape_img = tf.reshape(img, [3, 32, 32])

    transpose_img = tf.transpose(reshape_img, [1, 2, 0])
    cast_img = tf.cast(transpose_img, tf.float32)

    label_batch, img_batch = tf.train.batch([onehot_label, cast_img], batch_size=BATCH_SIZE, num_threads=1, capacity=BATCH_SIZE)
    return label_batch, img_batch


def get_y_predict(x, keep_prob):
    conv1 = conv_layer(x, shape=[3, 3, 3, 32])
    conv1_pool = max_pool_2x2(conv1)

    conv2 = conv_layer(conv1_pool, shape=[3, 3, 32, 64])
    conv2_pool = max_pool_2x2(conv2)
    conv2_flat = tf.reshape(conv2_pool, [-1, 8*8*64])

    full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))
    full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

    y_predict = full_layer(full1_drop, 10)
    return y_predict


if __name__ == '__main__':
    label_batch, img_batch = read_picture()

    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y = tf.placeholder(tf.float32, shape=[None, 10])

    keep_prob = tf.placeholder(tf.float32)

    y_predict = get_y_predict(x, keep_prob)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict, labels=y))
    train = tf.train.AdamOptimizer(1e-4).minimize(loss)

    correct = tf.equal(tf.argmax(y_predict, axis=1), tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(STEPS):
            labels, imgs = sess.run([label_batch, img_batch])
            if i%100 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={x:imgs, y:labels, keep_prob:1.0})
                print("step {}, training accuracy {}".format(i, train_accuracy))
            sess.run(train, feed_dict={x: imgs, y: labels, keep_prob: 0.8})

        coord.request_stop()
        coord.join(threads)

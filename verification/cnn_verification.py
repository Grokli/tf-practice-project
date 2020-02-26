import os
import numpy as np
import pandas as pd
import tensorflow as tf
from layers import conv_layer, max_pool_2x2, full_layer


DATA_PATH = './GenPics/'
BATCH_SIZE = 100
STEPS = 1000


def read_picture():
    file_list = os.listdir(DATA_PATH)
    file_list = [os.path.join(DATA_PATH, i) for i in file_list if i[-3:] == 'jpg']

    file_queue = tf.train.string_input_producer(file_list)

    reader = tf.WholeFileReader()
    filename, image = reader.read(file_queue)

    decode_image = tf.image.decode_jpeg(image)
    decode_image.set_shape([20, 80, 3])

    cast_image = tf.cast(decode_image, tf.float32)

    filename_batch, img_batch = tf.train.batch([filename, cast_image], batch_size=BATCH_SIZE, num_threads=1, capacity=BATCH_SIZE)

    return filename_batch, img_batch

def parse_csv():
    csv_data = pd.read_csv(DATA_PATH+'labels.csv', names=['file_num', 'chars'], index_col='file_num')
    csv_data['labels'] = csv_data['chars'].apply(lambda x:[ord(i)-ord('A') for i in x])
    return csv_data

def filename_to_label(filenames, csv_data):
    labels = []
    for filename in filenames:
        digit_str = ''.join(list(filter(str.isdigit, str(filename))))
        label = csv_data.loc[int(digit_str), 'labels']
        labels.append(label)
    return np.array(labels)

def get_y_predict(x, keep_prob):
    conv1 = conv_layer(x, shape=[3, 3, 3, 32])
    conv1_pool = max_pool_2x2(conv1)

    conv2 = conv_layer(conv1_pool, shape=[3, 3, 32, 64])
    conv2_pool = max_pool_2x2(conv2)
    conv2_flat = tf.reshape(conv2_pool, shape=[-1, 5*20*64])

    full1 = tf.nn.relu(full_layer(conv2_flat, 1024))
    full1_drop = tf.nn.dropout(full1, keep_prob=keep_prob)

    y_predict = full_layer(full1_drop, 26*4)

    return y_predict

if __name__ == '__main__':
    filename_batch, img_batch = read_picture()
    csv_data = parse_csv()

    x = tf.placeholder(tf.float32, shape=[None, 20, 80, 3])
    y = tf.placeholder(tf.float32, shape=[None, 26*4])

    keep_prob = tf.placeholder(tf.float32)
    y_predict = get_y_predict(x, keep_prob)

    entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_predict)
    loss = tf.reduce_mean(entropy)

    train = tf.train.AdamOptimizer(0.001).minimize(loss)

    correct = tf.reduce_all(tf.equal(tf.argmax(tf.reshape(y_predict, [-1, 4, 26]) ,axis=2), tf.argmax(tf.reshape(y, [-1, 4, 26]), axis=2)))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(STEPS):
            filenames, imgs = sess.run([filename_batch, img_batch])
            labels = filename_to_label(filenames, csv_data)
            labels_reshape = tf.reshape(tf.one_hot(labels, depth=26), shape=[-1, 26*4]).eval()
            if i%100 == 0:
                train_accuracy = sess.run(accuracy, feed_dict={x:imgs, y:labels_reshape, keep_prob:1.0})
                print("step {}, training accuracy {}".format(i, train_accuracy))
            sess.run(train, feed_dict={x: imgs, y: labels_reshape, keep_prob: 0.8})
        coord.request_stop()
        coord.join(threads)

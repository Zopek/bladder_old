import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
# from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
import time
import random

import read_data

filepath = '/DATA/data/qyzheng/lesion_one_normal_4'
# mnist = load_all_sets(filepath)

# Hyperparameter
growth_k = 24
nb_block = 2 # how many (dense block + Transition Layer) ?
init_learning_rate = 1e-4
epsilon = 1e-8 # AdamOptimizer epsilon
dropout_rate = 0.2
# dropout_rate = 0.4

# Momentum Optimizer will use
nesterov_momentum = 0.9
weight_decay = 1e-4

# Label & batch_size
class_num = 4
batch_size = 32

total_epochs = 100

height = 160
width = 160

def conv_layer(input, filter, kernel, stride=1, layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, filters=filter, kernel_size=kernel, strides=stride, padding='SAME')
        return network

def Global_Average_Pooling(x, stride=1):
    """
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride) # The stride value does not matter
    It is global average pooling without tflearn
    """

    return global_avg_pool(x, name='Global_avg_pooling')
    # But maybe you need to install h5py and curses or not


def Batch_Normalization(x, training, scope):
    with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))

def Drop_out(x, rate, training) :
    return tf.layers.dropout(inputs=x, rate=rate, training=training)

def Relu(x):
    return tf.nn.relu(x)

def Average_pooling(x, pool_size=[2,2], stride=2, padding='VALID'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)


def Max_Pooling(x, pool_size=[3,3], stride=2, padding='VALID'):
    return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x) :
    return tf.layers.dense(inputs=x, units=class_num, name='linear')



class DenseNet():
    def __init__(self, x, nb_blocks, filters, training):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training
        self.model = self.Dense_net(x)


    def bottleneck_layer(self, x, scope):
        # print(x)
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = conv_layer(x, filter=4 * self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            x = conv_layer(x, filter=self.filters, kernel=[3,3], layer_name=scope+'_conv2')
            x = Drop_out(x, rate=dropout_rate, training=self.training)

            # print(x)

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            shape = x.get_shape().as_list()
            in_channel = shape[3]
            # x = conv_layer(x, filter=self.filters, kernel=[1,1], layer_name=scope+'_conv1')
            x = conv_layer(x, filter=in_channel*0.5, kernel=[1,1], layer_name=scope+'_conv1')
            x = Drop_out(x, rate=dropout_rate, training=self.training)
            x = Average_pooling(x, pool_size=[2,2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = Concatenation(layers_concat)

            return x

    def Dense_net(self, input_x):
        x = conv_layer(input_x, filter=2 * self.filters, kernel=[7,7], stride=2, layer_name='conv0')
        x = Max_Pooling(x, pool_size=[3,3], stride=2)


        # """
        # for i in range(self.nb_blocks) :
        #     # 6 -> 12 -> 48
        #     x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_'+str(i))
        #     x = self.transition_layer(x, scope='trans_'+str(i))
        #
        #
        # """
        x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')
        x = self.transition_layer(x, scope='trans_1')

        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')
        x = self.transition_layer(x, scope='trans_2')

        x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3')
        x = self.transition_layer(x, scope='trans_3')
        

        x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_final')

        # 100 Layer
        x = Batch_Normalization(x, training=self.training, scope='linear_batch')
        x = Relu(x)
        x = Global_Average_Pooling(x)
        x = flatten(x)
        x = Linear(x)


        # x = tf.reshape(x, [-1, 10])
        return x

###======================== LOAD DATA ===================================###
## by importing this, you can load a training set and a validation set.
train_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/from_senior/0_cv_train.csv'
val_path = '/DB/rhome/qyzheng/Desktop/qyzheng/source/renji_data/from_senior/0_cv_val.csv'
train = []
test = []
train_size = 0
test_size = 0

with open(train_path, 'r') as f:
    reader = csv.reader(f)
    for i in reader:

        train.append(i)
        train_size += 1

with open(val_path, 'r') as f:
    reader = csv.reader(f)
    for i in reader:

        test.append(i)
        test_size += 1

train_epoch = train_size / batch_size
test_epoch = test_size / batch_size

###======================== Model ===================================###
x = tf.placeholder(tf.float32, shape=[None, height * width])
batch_images = tf.reshape(x, [-1, height, width, 3])

label = tf.placeholder(tf.float32, shape=[None, 3])

training_flag = tf.placeholder(tf.bool)


learning_rate = tf.placeholder(tf.float32, name='learning_rate')

logits = DenseNet(x=batch_images, nb_blocks=nb_block, filters=growth_k, training=training_flag).model
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

"""
l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=nesterov_momentum, use_nesterov=True)
train = optimizer.minimize(cost + l2_loss * weight_decay)
In paper, use MomentumOptimizer
init_learning_rate = 0.1
but, I'll use AdamOptimizer
"""

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=epsilon)
train = optimizer.minimize(cost)


correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
"""
correct_prediction_two = tf.equal(tf.argmax(logits[:, 4:7], 1), tf.argmax(label[:, 4:7], 1))
correct_prediction_thr = tf.equal(tf.argmax(logits[:, 7:11], 1), tf.argmax(label[:, 7:11], 1))
correct_prediction_fou = tf.equal(tf.argmax(logits[:, 11:14], 1), tf.argmax(label[:, 11:14], 1))
"""
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.summary.scalar('loss', cost)
tf.summary.scalar('accuracy', accuracy)

saver = tf.train.Saver(tf.global_variables())

"""
# delete?
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True 
session = tf.Session(config=config)
"""
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('./model')
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs', sess.graph)

    global_step = 0
    epoch_learning_rate = init_learning_rate
    for epoch in range(total_epochs):
        if epoch == (total_epochs * 0.5) or epoch == (total_epochs * 0.75):
            epoch_learning_rate = epoch_learning_rate / 10

        # total_batch = int(mnist.train.num_examples / batch_size)
        random.shuffle(train)
        random.shuffle(test)

        for step in range(train_epoch):
            # start_time = time.time()
            batch_x, batch_y = read_data.next_batch(train, batch_size, height, width)
            # batch_x, batch_y = mnist.train.next_batch(batch_size)
            # print('load time is : ', time.time() - start_time)
            train_feed_dict = {
                x: batch_x,
                label: batch_y,
                learning_rate: epoch_learning_rate,
                training_flag : True
            }

            _, loss = sess.run([train, cost], feed_dict=train_feed_dict)

            if step % 50 == 0:
                global_step += 50
                train_summary, train_accuracy = sess.run([merged, accuracy], feed_dict=train_feed_dict)
                # accuracy.eval(feed_dict=feed_dict)
                print("Step:", step, "Loss:", loss, "Training accuracy:", train_accuracy)
                writer.add_summary(train_summary, global_step=epoch)

            """
            if step  == total_train_batch - 1:
                # test_time = time.time()
                test_images, test_labels = input_data.test_next_batch(filepath, 1)
                test_feed_dict = {
                    x: test_images,
                    label: test_labels,
                    learning_rate: epoch_learning_rate,
                    training_flag : False
                }
                # print('test time is : ', time.time() - test_time)
            """
            # print('step time is : ', time.time() - start_time)

        for step in range(test_epoch):
            test_images, test_labels = read_data.next_batch(test, batch_size, height, width)
            test_feed_dict = {
                x: test_images,
                label: test_labels,
                learning_rate: epoch_learning_rate,
                training_flag : False
            }
            if step == 0:
                accuracy_rates = sess.run(accuracy, feed_dict=test_feed_dict)
            else:
                accuracy_rates += sess.run(accuracy, feed_dict=test_feed_dict)

            if step == total_test_batch - 1:
                print('Epoch:', '%04d' % (epoch + 1), '/ Test Accuracy =', accuracy_rates/total_test_batch)
            """
            if step % 5 == 0:
                print("Step:", step, "Test accuracy:", accuracy_rates/(step + 1))
            """
        # print('Epoch:', '%04d' % (epoch + 1), '/ Test Accuracy =', accuracy_rates/total_test_batch)

        # writer.add_summary(test_summary, global_step=epoch)

        """
        test_images, test_labels = input_data.test_next_batch(filepath, 2)
        test_feed_dict = {
            x: test_images,
            label: test_labels,
            learning_rate: epoch_learning_rate,
            training_flag : False
        }
        accuracy_rates = sess.run(accuracy, feed_dict=test_feed_dict)
        print('Epoch:', '%04d' % (epoch + 1), '/ Test2 Accuracy =', accuracy_rates)
        """

    saver.save(sess=sess, save_path='./model/dense.ckpt')
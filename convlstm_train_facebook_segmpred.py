from __future__ import absolute_import

import tensorflow as tf
import os

from videopred.modelloader.convlstm import BasicConvLSTMCell
from videopred.dataloader.vpn_minst import GenerateMovingMnistData
from videopred.dataloader.facebook_segmpred import GenerateFaceBookSegmPredData
from videopred.dataloader.config import vpn_mnist_config
from videopred.dataloader.config import facebook_segmpred_config
from videopred.modelloader.convlstm.layer_def import *

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', facebook_segmpred_config.batch_size, """the training batch size""")
tf.app.flags.DEFINE_integer('seq_length', 10, """the seq length""")
tf.app.flags.DEFINE_integer('seq_start', 5, """the seq start""")
tf.app.flags.DEFINE_float('lr', .001, """training learning rate""")
tf.app.flags.DEFINE_integer('max_step', 200000, """max step""")
tf.app.flags.DEFINE_string('train_dir', './checkpoints/train_store_conv_lstm', """train store dir""")

def conv_layer_relu(inputs, kernel_size, stride, num_features, idx):
    conv = conv_layer(inputs, kernel_size, stride, num_features, idx)
    return tf.nn.relu(conv)

def transpose_conv_layer_relu(inputs, kernel_size, stride, num_features, idx):
    conv = transpose_conv_layer(inputs, kernel_size, stride, num_features, idx)
    return tf.nn.relu(conv)

def transpose_conv_layer(inputs, kernel_size, stride, num_features, idx):
    with tf.variable_scope('{0}_conv'.format(idx)) as scope:
        inputs_channel = inputs.get_shape()[3]
        weights = tf.get_variable('weights', shape=(kernel_size, kernel_size, num_features, inputs_channel), initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases', shape=(num_features), initializer=tf.contrib.layers.xavier_initializer_conv2d())
        output_shape = tf.stack((tf.shape(inputs)[0], tf.shape(inputs)[1]*stride, tf.shape(inputs)[2]*stride, num_features))
        conv = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides=[1, stride, stride, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, biases)
        conv = tf.reshape(conv, (tf.shape(inputs)[0], tf.shape(inputs)[1]*stride, tf.shape(inputs)[2]*stride, num_features))
        return conv


def conv_layer(inputs, kernel_size, stride, num_features, idx):
    with tf.variable_scope('{0}_conv'.format(idx)) as scope:
        inputs_channel = inputs.get_shape()[3]
        weights = tf.get_variable('weights', shape=(kernel_size, kernel_size, inputs_channel, num_features), initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.get_variable('biases', shape=(num_features), initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, biases)
        return conv
    pass



def basic_lstm_network(inputs, hidden):
    conv1 = conv_layer_relu(inputs, kernel_size=3, stride=2, num_features=8, idx='encode_1')
    conv2 = conv_layer_relu(conv1, kernel_size=3, stride=1, num_features=8, idx='encode_2')
    conv3 = conv_layer_relu(conv2, kernel_size=3, stride=2, num_features=8, idx='encode_3')
    conv4 = conv_layer_relu(conv3, kernel_size=1, stride=1, num_features=4, idx='encode_4')
    conv_lstm_input = conv4
    with tf.variable_scope('conv_lstm', initializer=tf.random_uniform_initializer(-0.1, 0.1)):
        cell = BasicConvLSTMCell.BasicConvLSTMCell(shape=(16, 16), filter_size=(3,3), num_features=4)
        if hidden is None:
            hidden = cell.zero_state(FLAGS.batch_size, tf.float32)
        conv_lstm_output, hidden = cell(conv_lstm_input, hidden)
    conv5 = transpose_conv_layer_relu(conv_lstm_output, kernel_size=1, stride=1, num_features=8, idx='decode_5')
    conv6 = transpose_conv_layer_relu(conv5, kernel_size=3, stride=2, num_features=8, idx='decode_6')
    conv7 = transpose_conv_layer_relu(conv6, kernel_size=3, stride=1, num_features=8, idx='decode_7')
    # print(inputs.get_shape()[3])
    outputs = transpose_conv_layer(conv7, kernel_size=3, stride=2, num_features=int(inputs.get_shape()[3]), idx='decode_8')
    return outputs, hidden

basic_lstm_network_template = tf.make_template('basic_lstm_network', basic_lstm_network)


def residual_u_network(inputs, hiddens=None, start_filter_size=16, nr_downsamples=3, nr_residual_per_downsample=1, nonlinearity="relu"):

  # set filter size (after each down sample the filter size is doubled)
  filter_size = start_filter_size

  # set nonlinearity
  nonlinearity = set_nonlinearity(nonlinearity)

  # make list of hiddens if None
  if hiddens is None:
    hiddens = (2*nr_downsamples -1)*[None]

  # store for u network connections and new hiddens
  a = []
  hidden_out = []

  # encoding piece
  x_i = inputs
  for i in xrange(nr_downsamples):
    x_i = res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity, stride=2, name="res_encode_" + str(i) + "_block_0", begin_nonlinearity=False)
    for j in xrange(nr_residual_per_downsample - 1):
      x_i = res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity, name="res_encode_" + str(i) + "_block_" + str(j+1), begin_nonlinearity=True)
    x_i, hidden_new = res_block_lstm(x_i, hiddens[i], name="res_encode_lstm_" + str(i))
    a.append(x_i)
    hidden_out.append(hidden_new)
    filter_size = filter_size * 2

  # pop off last element to a.
  a.pop()
  filter_size = filter_size / 2

  # decoding piece
  for i in xrange(nr_downsamples - 1):
    filter_size = filter_size / 2
    x_i = transpose_conv_layer(x_i, 4, 2, filter_size, "up_conv_" + str(i))
    for j in xrange(nr_residual_per_downsample):
      x_i = res_block(x_i, a=a.pop(), filter_size=filter_size, nonlinearity=nonlinearity, name="res_decode_" + str(i) + "_block_" + str(j+1), begin_nonlinearity=True)
    x_i, hidden_new = res_block_lstm(x_i, hiddens[i + nr_downsamples], name="res_decode_lstm_" + str(i))
    hidden_out.append(hidden_new)

  x_i = transpose_conv_layer(x_i, 4, 2, int(inputs.get_shape()[-1]), "up_conv_" + str(nr_downsamples-1))

  return x_i, hidden_out

# make template for reuse
residual_u_network_template = tf.make_template('residual_u_network', residual_u_network)

def train():
    with tf.Graph().as_default():
        data_generate = GenerateFaceBookSegmPredData()
        # data_generate = GenerateMovingMnistData()
        FLAGS.seq_length = data_generate.num_timestamps
        FLAGS.seq_start = data_generate.num_timestamps - 1
        x = tf.placeholder(dtype=tf.float32, shape=(FLAGS.batch_size, FLAGS.seq_length, data_generate.height, data_generate.width, data_generate.channel))
        # print(x.shape)
        x_dropout = tf.nn.dropout(x, keep_prob=0.5)
        hidden = None
        x_unwrap = []
        for i in range(FLAGS.seq_length):
            # print i
            if i < FLAGS.seq_start:
                outputs, hidden = basic_lstm_network_template(x_dropout[:, i, :, :, :], hidden)
                # outputs, hidden = residual_u_network_template(x[:, i, :, :, :], hidden)
            else:
                outputs, hidden = basic_lstm_network_template(outputs, hidden)
                # outputs, hidden = residual_u_network_template(outputs, hidden)
            x_unwrap.append(outputs)
        x_unwrap = tf.stack(x_unwrap)
        x_unwrap = tf.transpose(x_unwrap, (1,0,2,3,4))
        #  print(x_unwrap.shape)
        loss = tf.nn.l2_loss(x[:,FLAGS.seq_start:,:,:,:] - x_unwrap[:,FLAGS.seq_start:,:,:,:])
        tf.summary.scalar('loss', loss)
        train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()


        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        graph_def = sess.graph.as_graph_def(add_shapes=True)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph_def=graph_def)

        loss_sum = 0.0
        step_num = 100
        save_num = 5000

        for step in range(FLAGS.max_step):
            dataset_batch = data_generate.next_batch()
            dataset_batch = dataset_batch / 19.0
            #  print(dataset_batch.shape)
            train_op_r, loss_r = sess.run([train_op, loss], feed_dict={x:dataset_batch})
            # print(loss_r)
            loss_sum += loss_r
            if step%step_num==0 and step != 0:
                loss_avg = loss_sum / (step_num+1)
                loss_sum = 0.0
                summary_str = sess.run(summary_op, feed_dict={x:dataset_batch})
                summary_writer.add_summary(summary_str, step)
                # print(loss_r)
                print('loss_avg:', loss_avg)
            elif step%save_num==0 and step != 0:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):
    print('convlstm----main----in----')
    print(argv)
    train()
    print('convlstm----main----out----')

if __name__ == '__main__':
    tf.app.run()


#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np
import time

from videopred.dataloader.facebook_segmpred import GenerateFaceBookSegmPredData
from videopred.dataloader.vpn_minst import GenerateMovingMnistData
from videopred.dataloader.config import facebook_segmpred_config

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('weight_decay', 0.0005,
                          """ """)
tf.app.flags.DEFINE_string('train_dir', './checkpoints/train_store_conv_lstm',
                            """dir to store trained net""")
tf.app.flags.DEFINE_integer('seq_length', 10,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('seq_start', 5,
                            """ start of seq generation""")
tf.app.flags.DEFINE_integer('max_step', 200000,
                            """max num of steps""")
tf.app.flags.DEFINE_float('keep_prob', .8,
                            """for dropout""")
tf.app.flags.DEFINE_float('lr', .001,
                            """for dropout""")
tf.app.flags.DEFINE_integer('batch_size', facebook_segmpred_config.batch_size,
                            """batch size for training""")
tf.app.flags.DEFINE_float('weight_init', .1,
                            """weight init for fully connected layers""")
tf.app.flags.DEFINE_string('model', 'residual_u_network',
                            """ model to train """)


def int_shape(x):
    return list(map(int, x.get_shape()))


def concat_elu(x):
    """ like concatenated ReLU (http://arxiv.org/abs/1603.05201), but then with ELU """
    axis = len(x.get_shape()) - 1
    return tf.nn.elu(tf.concat([x, -x], axis))


def set_nonlinearity(name):
    if name == 'concat_elu':
        return concat_elu
    elif name == 'elu':
        return tf.nn.elu
    elif name == 'concat_relu':
        return tf.nn.crelu
    elif name == 'relu':
        return tf.nn.relu
    else:
        raise ('nonlinearity ' + name + ' is not supported')


def _variable(name, shape, initializer):
    """Helper to create a Variable.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    # getting rid of stddev for xavier ## testing this for faster convergence
    var = tf.get_variable(name, shape, initializer=initializer)
    return var


def conv_layer(inputs, kernel_size, stride, num_features, idx, nonlinearity=None):
    with tf.variable_scope('{0}_conv'.format(idx)) as scope:
        input_channels = inputs.get_shape()[3]

        weights = _variable('weights', shape=[kernel_size, kernel_size, input_channels, num_features],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = _variable('biases', [num_features], initializer=tf.contrib.layers.xavier_initializer_conv2d())

        conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, biases)
        if nonlinearity is not None:
            conv = nonlinearity(conv)
        return conv


def transpose_conv_layer(inputs, kernel_size, stride, num_features, idx, nonlinearity=None):
    with tf.variable_scope('{0}_trans_conv'.format(idx)) as scope:
        input_channels = inputs.get_shape()[3]

        weights = _variable('weights', shape=[kernel_size, kernel_size, num_features, input_channels],
                            initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = _variable('biases', [num_features], initializer=tf.contrib.layers.xavier_initializer_conv2d())
        batch_size = tf.shape(inputs)[0]
        output_shape = tf.stack(
            [tf.shape(inputs)[0], tf.shape(inputs)[1] * stride, tf.shape(inputs)[2] * stride, num_features])
        conv = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides=[1, stride, stride, 1], padding='SAME')
        conv = tf.nn.bias_add(conv, biases)
        if nonlinearity is not None:
            conv = nonlinearity(conv)
        shape = int_shape(inputs)
        conv = tf.reshape(conv, [shape[0], shape[1] * stride, shape[2] * stride, num_features])
        return conv


def fc_layer(inputs, hiddens, idx, nonlinearity=None, flat=False):
    with tf.variable_scope('{0}_fc'.format(idx)) as scope:
        input_shape = inputs.get_shape().as_list()
        if flat:
            dim = input_shape[1] * input_shape[2] * input_shape[3]
            inputs_processed = tf.reshape(inputs, [-1, dim])
        else:
            dim = input_shape[1]
            inputs_processed = inputs

        weights = _variable('weights', shape=[dim, hiddens], initializer=tf.contrib.layers.xavier_initializer())
        biases = _variable('biases', [hiddens], initializer=tf.contrib.layers.xavier_initializer())
        output_biased = tf.add(tf.matmul(inputs_processed, weights), biases, name=str(idx) + '_fc')
        if nonlinearity is not None:
            output_biased = nonlinearity(ouput_biased)
        return output_biased


def nin(x, num_units, idx):
    """ a network in network layer (1x1 CONV) """
    s = int_shape(x)
    x = tf.reshape(x, [np.prod(s[:-1]), s[-1]])
    x = fc_layer(x, num_units, idx)
    return tf.reshape(x, s[:-1] + [num_units])


def res_block(x, a=None, filter_size=16, nonlinearity=concat_elu, keep_p=1.0, stride=1, gated=True, name="resnet",
              begin_nonlinearity=True):
    orig_x = x
    if begin_nonlinearity:
        x = nonlinearity(x)
    if stride == 1:
        x = conv_layer(x, 3, stride, filter_size, name + '_conv_1')
    elif stride == 2:
        x = conv_layer(x, 4, stride, filter_size, name + '_conv_1')
    else:
        print("stride > 2 is not supported")
        exit()
    if a is not None:
        shape_a = int_shape(a)
        shape_x_1 = int_shape(x)
        a = tf.pad(
            a, [[0, 0], [0, shape_x_1[1] - shape_a[1]], [0, shape_x_1[2] - shape_a[2]],
                [0, 0]])
        x += nin(nonlinearity(a), filter_size, name + '_nin')
    x = nonlinearity(x)
    if keep_p < 1.0:
        x = tf.nn.dropout(x, keep_prob=keep_p)
    if not gated:
        x = conv_layer(x, 3, 1, filter_size, name + '_conv_2')
    else:
        x = conv_layer(x, 3, 1, filter_size * 2, name + '_conv_2')
        x_1, x_2 = tf.split(x, 2, 3)
        x = x_1 * tf.nn.sigmoid(x_2)

    if int(orig_x.get_shape()[2]) > int(x.get_shape()[2]):
        assert (int(orig_x.get_shape()[2]) == 2 * int(x.get_shape()[2]), "res net block only supports stirde 2")
        orig_x = tf.nn.avg_pool(orig_x, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    # pad it
    out_filter = filter_size
    in_filter = int(orig_x.get_shape()[-1])
    if out_filter > in_filter:
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter - in_filter), 0]])
    elif out_filter < in_filter:
        orig_x = nin(orig_x, out_filter, name + '_nin_pad')
    return orig_x + x


def res_block_lstm(x, hidden_state=None, keep_p=1.0, name="resnet_lstm"):
    orig_x = x
    if hidden_state is not None:
        hidden_state_1 = hidden_state[0]
        hidden_state_2 = hidden_state[1]
    else:
        hidden_state_1 = None
        hidden_state_2 = None
    filter_size = orig_x.get_shape().as_list()[-1]

    with tf.variable_scope(name + "_conv_LSTM_1", initializer=tf.random_uniform_initializer(-0.01, 0.01)) as scope:
        lstm_cell_1 = BasicConvLSTMCell([int(x.get_shape()[1]), int(x.get_shape()[2])], [3, 3],
                                                          filter_size)
        if hidden_state_1 == None:
            batch_size = x.get_shape()[0]
            hidden_state_1 = lstm_cell_1.zero_state(batch_size, tf.float32)
        x_1, hidden_state_1 = lstm_cell_1(x, hidden_state_1, scope=scope)

    if keep_p < 1.0:
        x_1 = tf.nn.dropout(x_1, keep_prob=keep_p)

    with tf.variable_scope(name + "_conv_LSTM_2", initializer=tf.random_uniform_initializer(-0.01, 0.01)) as scope:
        lstm_cell_2 = BasicConvLSTMCell([int(x_1.get_shape()[1]), int(x_1.get_shape()[2])], [3, 3],
                                                          filter_size)
        if hidden_state_2 == None:
            batch_size = x_1.get_shape()[0]
            hidden_state_2 = lstm_cell_2.zero_state(batch_size, tf.float32)
        x_2, hidden_state_2 = lstm_cell_2(x_1, hidden_state_2, scope=scope)

    return orig_x + x_2, [hidden_state_1, hidden_state_2]

class ConvRNNCell(object):
    """Abstract object representing an Convolutional RNN cell.
    """

    def __call__(self, inputs, state, scope=None):
        """Run this RNN cell on inputs, starting from the given state.
        """
        raise NotImplementedError("Abstract method")

    @property
    def state_size(self):
        """size(s) of state(s) used by this cell.
        """
        raise NotImplementedError("Abstract method")

    @property
    def output_size(self):
        """Integer or TensorShape: size of outputs produced by this cell."""
        raise NotImplementedError("Abstract method")

    def zero_state(self, batch_size, dtype):
        """Return zero-filled state tensor(s).
        Args:
          batch_size: int, float, or unit Tensor representing the batch size.
          dtype: the data type to use for the state.
        Returns:
          tensor of shape '[batch_size x shape[0] x shape[1] x num_features]
          filled with zeros
        """

        shape = self.shape
        num_features = self.num_features
        zeros = tf.zeros([batch_size, shape[0], shape[1], num_features * 2])
        return zeros


class BasicConvLSTMCell(ConvRNNCell):
    """Basic Conv LSTM recurrent network cell. The
    """

    def __init__(self, shape, filter_size, num_features, forget_bias=1.0, input_size=None,
                 state_is_tuple=False, activation=tf.nn.tanh):
        """Initialize the basic Conv LSTM cell.
        Args:
          shape: int tuple thats the height and width of the cell
          filter_size: int tuple thats the height and width of the filter
          num_features: int thats the depth of the cell 
          forget_bias: float, The bias added to forget gates (see above).
          input_size: Deprecated and unused.
          state_is_tuple: If True, accepted and returned states are 2-tuples of
            the `c_state` and `m_state`.  If False, they are concatenated
            along the column axis.  The latter behavior will soon be deprecated.
          activation: Activation function of the inner states.
        """
        # if not state_is_tuple:
        # logging.warn("%s: Using a concatenated state is slower and will soon be "
        #             "deprecated.  Use state_is_tuple=True.", self)
        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)
        self.shape = shape
        self.filter_size = filter_size
        self.num_features = num_features
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_units, self._num_units)
                if self._state_is_tuple else 2 * self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
            # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                c, h = tf.split(axis=3, num_or_size_splits=2, value=state)
            concat = _conv_linear([inputs, h], self.filter_size, self.num_features * 4, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(axis=3, num_or_size_splits=4, value=concat)

            new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) *
                     self._activation(j))
            new_h = self._activation(new_c) * tf.nn.sigmoid(o)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                new_state = tf.concat(axis=3, values=[new_c, new_h])
            return new_h, new_state


def _conv_linear(args, filter_size, num_features, bias, bias_start=0.0, scope=None):
    """convolution:
    Args:
      args: a 4D Tensor or a list of 4D, batch x n, Tensors.
      filter_size: int tuple of filter height and width.
      num_features: int, number of features.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".
    Returns:
      A 4D Tensor with shape [batch h w num_features]
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """

    # Calculate the total size of arguments on dimension 1.
    total_arg_size_depth = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 4:
            raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
        if not shape[3]:
            raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
        else:
            total_arg_size_depth += shape[3]

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope or "Conv"):
        matrix = tf.get_variable(
            "Matrix", [filter_size[0], filter_size[1], total_arg_size_depth, num_features], dtype=dtype)
        if len(args) == 1:
            res = tf.nn.conv2d(args[0], matrix, strides=[1, 1, 1, 1], padding='SAME')
        else:
            res = tf.nn.conv2d(tf.concat(axis=3, values=args), matrix, strides=[1, 1, 1, 1], padding='SAME')
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [num_features],
            dtype=dtype,
            initializer=tf.constant_initializer(
                bias_start, dtype=dtype))
    return res + bias_term


def select_network(name="residual_u_network"):
    network = None
    if name == "basic_network":
        network = basic_network_template
    elif name == "residual_u_network":
        network = residual_u_network_template
    return network


def basic_network(inputs, hidden, lstm=True):
    conv1 = conv_layer(inputs, 3, 2, 8, "encode_1", nonlinearity=tf.nn.elu)
    # conv2
    conv2 = conv_layer(conv1, 3, 1, 8, "encode_2", nonlinearity=tf.nn.elu)
    # conv3
    conv3 = conv_layer(conv2, 3, 2, 8, "encode_3", nonlinearity=tf.nn.elu)
    # conv4
    conv4 = conv_layer(conv3, 1, 1, 4, "encode_4", nonlinearity=tf.nn.elu)
    y_0 = conv4
    if lstm:
        # conv lstm cell
        with tf.variable_scope('conv_lstm', initializer=tf.random_uniform_initializer(-.01, 0.1)):
            cell = BasicConvLSTMCell([8, 8], [3, 3], 4)
            if hidden is None:
                hidden = cell.zero_state(FLAGS.batch_size, tf.float32)
            y_1, hidden = cell(y_0, hidden)
    else:
        y_1 = conv_layer(y_0, 3, 1, 8, "encode_3", nonlinearity=tf.nn.elu)

    # conv5
    conv5 = transpose_conv_layer(y_1, 1, 1, 8, "decode_5", nonlinearity=tf.nn.elu)
    # conv6
    conv6 = transpose_conv_layer(conv5, 3, 2, 8, "decode_6", nonlinearity=tf.nn.elu)
    # conv7
    conv7 = transpose_conv_layer(conv6, 3, 1, 8, "decode_7", nonlinearity=tf.nn.elu)
    # x_1
    x_1 = transpose_conv_layer(conv7, 3, 2, 3, "decode_8")  # set activation to linear
    # print('inputs:', inputs)
    # print('conv1:', conv1)
    # print('conv2:', conv2)
    # print('conv3:', conv3)
    # print('conv4:', conv4)
    # print('y_1:', y_1)
    # print('hidden:', hidden)
    # print('conv5:', conv5)
    # print('conv6:', conv6)
    # print('conv7:', conv7)
    # print('x_1:', x_1)
    # exit(0)

    return x_1, hidden


# make a template for reuse
basic_network_template = tf.make_template('basic_network', basic_network)


def residual_u_network(inputs, hiddens=None, start_filter_size=16, nr_downsamples=3, nr_residual_per_downsample=1,
                       nonlinearity="concat_elu"):
    # set filter size (after each down sample the filter size is doubled)
    filter_size = start_filter_size

    # set nonlinearity
    nonlinearity = set_nonlinearity(nonlinearity)

    # make list of hiddens if None
    if hiddens is None:
        hiddens = (2 * nr_downsamples - 1) * [None]

    # store for u network connections and new hiddens
    a = []
    hidden_out = []

    # encoding piece
    x_i = inputs
    for i in xrange(nr_downsamples):
        x_i = res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity, stride=2,
                        name="res_encode_" + str(i) + "_block_0", begin_nonlinearity=False)
        for j in xrange(nr_residual_per_downsample - 1):
            x_i = res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity,
                            name="res_encode_" + str(i) + "_block_" + str(j + 1), begin_nonlinearity=True)
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
            x_i = res_block(x_i, a=a.pop(), filter_size=filter_size, nonlinearity=nonlinearity,
                            name="res_decode_" + str(i) + "_block_" + str(j + 1), begin_nonlinearity=True)
        x_i, hidden_new = res_block_lstm(x_i, hiddens[i + nr_downsamples], name="res_decode_lstm_" + str(i))
        hidden_out.append(hidden_new)

    x_i = transpose_conv_layer(x_i, 4, 2, int(inputs.get_shape()[-1]), "up_conv_" + str(nr_downsamples - 1))

    return x_i, hidden_out


# make template for reuse
residual_u_network_template = tf.make_template('residual_u_network', residual_u_network)

def train():
    """Train ring_net for a number of steps."""
    with tf.Graph().as_default():
        # data_generate = GenerateFaceBookSegmPredData()
        data_generate = GenerateMovingMnistData()
        FLAGS.seq_start = data_generate.num_timestamps - 1
        # make inputs
        x = tf.placeholder(tf.float32, [FLAGS.batch_size, data_generate.num_timestamps, data_generate.height, data_generate.width, data_generate.channel])

        # possible dropout inside
        keep_prob = tf.placeholder("float")
        x_dropout = tf.nn.dropout(x, keep_prob)

        # create network
        network_template = select_network(FLAGS.model)
        x_unwrap = []

        # conv network
        hidden = None
        for i in xrange(data_generate.num_timestamps - 1):
            if i < FLAGS.seq_start:
                x_1, hidden = network_template(x_dropout[:, i, :, :, :], hidden)
            else:
                x_1, hidden = network_template(x_1, hidden)
            x_unwrap.append(x_1)

        # pack them all together
        x_unwrap = tf.stack(x_unwrap)
        x_unwrap = tf.transpose(x_unwrap, [1, 0, 2, 3, 4])

        # this part will be used for generating video
        x_unwrap_g = []
        hidden_g = None
        for i in xrange(50):
            if i < FLAGS.seq_start:
                x_1_g, hidden_g = network_template(x_dropout[:, i, :, :, :], hidden_g)
            else:
                x_1_g, hidden_g = network_template(x_1_g, hidden_g)
            x_unwrap_g.append(x_1_g)

        # pack them generated ones
        x_unwrap_g = tf.stack(x_unwrap_g)
        x_unwrap_g = tf.transpose(x_unwrap_g, [1, 0, 2, 3, 4])

        # calc total loss (compare x_t to x_t+1)
        # print(x.get_shape())
        # print(x_unwrap.get_shape())
        loss = tf.nn.l2_loss(x[:, FLAGS.seq_start + 1:, :, :, :] - x_unwrap[:, FLAGS.seq_start:, :, :, :])
        tf.summary.scalar('loss', loss)

        # training
        train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)

        # List of all Variables
        variables = tf.global_variables()

        # Build a saver
        saver = tf.train.Saver(tf.global_variables())

        # Summary op
        summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess = tf.Session()

        # init if this is the very time training
        print("init network from scratch")
        sess.run(init)

        # Summary op
        graph_def = sess.graph.as_graph_def(add_shapes=True)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph_def=graph_def)


        for step in xrange(FLAGS.max_step):
            dat = data_generate.next_batch()
            t = time.time()
            _, loss_r = sess.run([train_op, loss], feed_dict={x: dat, keep_prob: FLAGS.keep_prob})
            elapsed = time.time() - t

            print(loss_r)

            if step % 100 == 0 and step != 0:
                summary_str = sess.run(summary_op, feed_dict={x: dat, keep_prob: FLAGS.keep_prob})
                summary_writer.add_summary(summary_str, step)
                print("time per batch is " + str(elapsed))
                print(step)
                print(loss_r)

            assert not np.isnan(loss_r), 'Model diverged with loss = NaN'

def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

DEFAULT_PADDING = 'SAME'


def layer(op):
    '''Decorator for composable network layers.'''

    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.terminals) == 0:
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)
        # Perform the operation and get the output.
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        self.feed(layer_output)
        # Return self for chained calls.
        return self

    return layer_decorated


class Network(object):

    def __init__(self, inputs, trainable=True, initializer = None):
        # The input nodes for this network
        self.inputs = inputs
        # The current list of terminal nodes
        self.terminals = []
        # Mapping from layer names to layers
        self.layers = dict(inputs)
        # If true, the resulting variables are set as trainable
        self.trainable = trainable
        # Switch variable for dropout
        self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
                                                       shape=[],
                                                       name='use_dropout')
        if initializer is None:
            self.initializer =  tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        self.vars = []
        self.setup()

    def setup(self):
        '''Construct the network. '''
        raise NotImplementedError('Must be implemented by the subclass.')

    def load(self, data_path, session, ignore_missing=False):
        '''Load network weights.
        data_path: The path to the numpy-serialized network weights
        session: The current TensorFlow session
        ignore_missing: If true, serialized weights for missing layers are ignored.
        '''
        data_dict = np.load(data_path).item()
        self.load_from_dict(data_dict, ignore_missing, session)

    def load_from_dict(self, data_dict, ignore_missing, session):
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        '''Set the input(s) for the next operation by replacing the terminal nodes.
        The arguments can be either layer names or the actual layers.
        '''
        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, basestring):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        '''Returns the current network output.'''
        return self.terminals[-1]

    def get_unique_name(self, prefix):
        '''Returns an index-suffixed unique name for the given prefix.
        This is used for auto-generating layer names based on the type-prefix.
        '''
        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape, trainable=None, initializer=None):
        '''Creates a new TensorFlow variable.'''
        if trainable is None:
            trainable = self.trainable
        if initializer is None:
            initializer = self.initializer
        var = tf.get_variable(name, shape, trainable=trainable, initializer=initializer)
        self.vars.append(var)
        return var

    def validate_padding(self, padding):
        '''Verifies that the padding is one of the supported ones.'''
        assert padding in ('SAME', 'VALID')
        

#    @layer
#    def deconv(self, input, output_dim, name, ks=4, s=2, reuse=False):
#        with tf.variable_scope(name):
##            return slim.conv2d_transpose(input, output_dim, ks, s, padding='SAME', activation_fn=None,
#                                   weights_initializer=self.initializer,
#                                  biases_initializer=None)

    @layer
    def deconv(self,
               input,
               output_shape,
               k_h,
               k_w,
               s_h,
               s_w,
               name,
               relu=True,
               biased=True,
               trainable=True,
               reuse=False):

        with tf.variable_scope(name, reuse=reuse) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, output_shape[-1],input.get_shape()[-1]], trainable=trainable)

            output = tf.nn.conv2d_transpose(input, kernel, output_shape=output_shape,
                                        strides=[1, s_h, s_w, 1])

            if biased:
                biases = self.make_var('biases', [output_shape[-1]], trainable=trainable)
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
         
        return output


    @layer
    def conv(self,
             input,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding=DEFAULT_PADDING,
             group=1,
             biased=True,
             trainable=None, reuse=False):
        # Verify that the padding is acceptable
        self.validate_padding(padding)
        # Get the number of channels in the input
        c_i = input.get_shape()[-1]
        # Verify that the grouping parameter is valid
        assert c_i % group == 0
        assert c_o % group == 0
        # Convolution for a given input and kernel
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name, reuse=reuse) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o / group], trainable=trainable)
            if group == 1:
                # This is the common-case. Convolve the input without any further complications.
                output = convolve(input, kernel)
            else:
                # Split the input into groups and then convolve each of them independently
                input_groups = tf.split(input, group, axis=np.int32(3))
                output_groups = [convolve(i, kernel) for i in input_groups]
                # Concatenate the groups
                output = tf.concat(values=output_groups, axis=3)
            # Add the biases
            if biased:
                biases = self.make_var('biases', [c_o], trainable=trainable)
                output = tf.nn.bias_add(output, biases)
            if relu:
                # ReLU non-linearity
                output = tf.nn.relu(output, name=scope.name)
            return output

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def sigmoid(self, input, name):
        return tf.nn.sigmoid(input, name=name)


    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)


    @layer
    def sigmoid(self, input, name):
        return tf.nn.sigmoid(input, name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def lrn(self, input, radius, alpha, beta, name, bias=1.0):
        return tf.nn.local_response_normalization(input,
                                                  depth_radius=radius,
                                                  alpha=alpha,
                                                  beta=beta,
                                                  bias=bias,
                                                  name=name)

    @layer
    def concat(self, inputs, axis, name):
        return tf.concat( values=inputs, axis =axis, name=name)

    @layer
    def add(self, inputs, name):
        return tf.add_n(inputs, name=name)

    @layer
    def fc(self, input, num_out, name, relu=True, trainable=None, reuse=False):
        with tf.variable_scope(name, reuse=reuse) as scope:
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                # The input is spatial. Vectorize it first.
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(input, [-1, dim])
            else:
                feed_in, dim = (input, input_shape[-1].value)
            weights = self.make_var('weights', shape=[dim, num_out], trainable=trainable)
            biases = self.make_var('biases', [num_out], trainable=trainable)
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=scope.name)
            return fc

    @layer
    def softmax(self, input, name):
        input_shape = map(lambda v: v.value, input.get_shape())
        if len(input_shape) > 2:
            # For certain models (like NiN), the singleton spatial dimensions
            # need to be explicitly squeezed, since they're not broadcast-able
            # in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
            if input_shape[1] == 1 and input_shape[2] == 1:
                input = tf.squeeze(input, squeeze_dims=[1, 2])
            else:
                raise ValueError('Rank 2 tensor input expected for softmax!')
        return tf.nn.softmax(input, name=name)

    @layer
    def batch_normalization(self, input, name, scale_offset=True, relu=False, trainable=None, reuse=False):
        # NOTE: Currently, only inference is supported
        with tf.variable_scope(name, reuse=reuse) as scope:
            shape = [input.get_shape()[-1]]
            if scale_offset:
                scale = self.make_var('scale', shape=shape, trainable=trainable)
                offset = self.make_var('offset', shape=shape, trainable=trainable)
            else:
                scale, offset = (None, None)
            output = tf.nn.batch_normalization(
                input,
                mean=self.make_var('mean', shape=shape, trainable=trainable),
                variance=self.make_var('variance', shape=shape, trainable=trainable),
                offset=offset,
                scale=scale,
                # TODO: This is the default Caffe batch norm eps
                # Get the actual eps from parameters
                variance_epsilon=1e-5,
                name=name)
            if relu:
                output = tf.nn.relu(output)
            return output
        
    @layer
    def instance_norm(self, input, name, relu=False, trainable=None, reuse=False):
        with tf.variable_scope(name, reuse=reuse) as scope:
            depth = input.get_shape()[-1]
            scale = self.make_var("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32), trainable=trainable)
            offset = self.make_var("offset", [depth], initializer=tf.constant_initializer(0.0), trainable=trainable)
            mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
            epsilon = 1e-5
            inv = tf.rsqrt(variance + epsilon)
            normalized = (input-mean)*inv
            output = scale*normalized + offset
            if relu:
                output = tf.nn.relu(output)
            return output

    @layer
    def embedding(self, input, name, vocab_size, embedding_size):
        with tf.variable_scope(name) as scope:
            embedding_matrix = self.make_var('embedding', shape=[vocab_size, embedding_size])
            return tf.nn.embedding_lookup(embedding_matrix, input)

    @layer
    def dropout(self, input, keep_prob, name):
        keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
        return tf.nn.dropout(input, keep, name=name)

    @layer
    def tanh(self, input, name):
        return tf.nn.tanh(input, name=name)

    @layer
    def reshape(self, input, shape, name):
        return tf.reshape(input, shape=shape, name=name)


    @layer
    def spatial_softmax(self, input, name):
        """Spatial Softmax on convolutions. Produces x,y coordinates of expected activation of each filter.
        Assumes NWHC format"""
        batch_size, num_rows, num_cols, num_channels = [d.value for d in input.shape]
        if batch_size is None:
            batch_size = -1

        x_map = np.empty([num_rows, num_cols], np.float32)
        y_map = np.empty([num_rows, num_cols], np.float32)

        for i in range(num_rows):
            for j in range(num_cols):
                x_map[i, j] = (i - num_rows / 2.0) / num_rows
                y_map[i, j] = (j - num_cols / 2.0) / num_cols

        x_map = tf.convert_to_tensor(x_map)
        y_map = tf.convert_to_tensor(y_map)

        x_map = tf.reshape(x_map, [num_rows * num_cols])
        y_map = tf.reshape(y_map, [num_rows * num_cols])

        features = tf.reshape(tf.transpose(input, [0, 3, 1, 2]),
                              [batch_size, num_channels, num_rows * num_cols])
        softmax = tf.nn.softmax(features)

        fp_x = tf.reduce_sum(tf.multiply(x_map, softmax), 2)
        fp_y = tf.reduce_sum(tf.multiply(y_map, softmax), 2)

        fp = tf.reshape(tf.concat(axis=1, values=[fp_x, fp_y]), [batch_size, num_channels * 2], name=name)
        return fp

    @layer
    def depth_lookup(self, (input, depth_im), name):
        """ lookup from a depth image the location of the features of the features  """
        batch_size, num_rows, num_cols, num_channels = [d.value for d in input.shape]
        if batch_size is None:
            batch_size = -1

        features = tf.reshape(tf.transpose(input, [0, 3, 1, 2]),
                              [batch_size, num_channels, num_rows * num_cols])
        softmax = tf.nn.softmax(features)
        softmax_T = tf.transpose(softmax, [1,0,2]) ## Since softmax is batch, filter, num_row*num_col and depth is batch, num_row*num_col.. Move filter aside for a valid elementwise multiplication
        prod = tf.multiply(softmax_T, depth_im)
        prod_T = tf.transpose(prod, [1,0,2])
        return tf.reduce_sum(prod_T, axis=-1, name=name)

    def load_with_transformation(self, model, transforms):
        out = {}
        for key in model.keys():
            v = model[key]
            for transform in transforms:
                out_ky = transform(key)
                if out_ky not in out:
                    out[out_ky] = v
        return out
   
   
    def get_vars(self):
        return self.vars
 
    #TODO:  support for files with > 2 level deep variables
    def export(self, sess, file_name):
        s = []
        for layer in self.layers:
            s+=([ (layer, v) for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) if v.name.startswith(layer) ])
        outdict = {}
        for layer, var in s:
            varname = var.name.replace(':0', '')
            varvalue = var.eval(sess)
            if ('/' in varname) and (len(varname) == 2):
                (base, det) = varname.split('/')
                indict = outdict.get(base, {})
                indict[det] = varvalue
                outdict[base] = indict
            else:
                outdict[varname] = varvalue
        np.save(file_name, outdict)

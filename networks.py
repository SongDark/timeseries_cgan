import tensorflow as tf 
from utils import *

class Generator(BasicBlock):
    '''
        z: 2d tensor. [N, noise_dim]
        y: 2d tensor. [N, class_num]
        output_dim: scalar. channel of output
    Returns:
        net: 
    '''
    def __init__(self, output_dim, name=None):
        super(Generator, self).__init__(None, name or "G")
        self.output_dim = output_dim
    
    def __call__(self, z, y=None, is_training=True, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            batch_size = z.get_shape().as_list()[0]
            if y is not None:
                z = tf.concat([z,y], 1)

            # [bz, 1024]
            net = tf.nn.relu(bn(dense(z, 1024, name='g_fc1'), is_training, name='g_bn1'))
            net = tf.nn.relu(bn(dense(net, 128*46, name='g_fc2'), is_training, name='g_bn2'))
            net = tf.reshape(net, [batch_size, 46, 1, 128])
            # [bz, 92, 1, 64]
            net = tf.nn.relu(bn(deconv2d(net, 64, 4, 1, 2, 1, padding='SAME', name='g_dc3'), is_training, name='g_bn3'))
            # [bz, 184, 1, 32]
            net = deconv2d(net, 1, 4, 2, 2, 2, padding='SAME', name='g_dc4')
            # [bz, 182, 2, output_dim]
            net = conv2d(net, self.output_dim, 3, 1, 1, 1, padding='VALID', name='g_c5')
        return net

class Discriminator(BasicBlock):
    '''
        x: 4d tensor. [N, T, d, C]
        y: 2d tensor. [N, class_num]
        class_num: scalar.
    Returns:
        yd: 2d tensor. [N, 1]. 
        net: 2d tensor. [N, 1024].
        yc: 2d tensor. [N, class_num]
    '''
    def __init__(self, class_num=None, name=None):
        super(Discriminator, self).__init__(None, name or "D")
        self.class_num = class_num
    
    def __call__(self, x, y=None, is_training=True, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            batch_size = x.get_shape().as_list()[0]
            if y is not None:
                ydim = y.get_shape().as_list()[-1]
                y = tf.reshape(y, [batch_size, 1, 1, ydim])
                x = conv_cond_concat(x, y)
            # [bz, 91, 1, 32]
            net = lrelu(conv2d(x, 32, 4, 1, 2, 1, padding="SAME", name='d_c1'), name='d_l1')
            # [bz, 46, 1, 64]
            net = lrelu(bn(conv2d(net, 64, 4, 1, 2, 1, padding="SAME", name='d_c2'), is_training, name='d_bn2'), name='d_l2')
            net = tf.reshape(net, [batch_size, -1])
            # [bz, 1024]
            net = lrelu(bn(dense(net, 1024, name='d_fc3'), is_training, name='d_bn3'), name='d_l3')
            # [bz, 1]
            yd = dense(net, 1, name='D_dense')
            if self.class_num:
                yc = dense(net, self.class_num, name='C_dense')
                return yd, net, yc 
            else:
                return yd, net
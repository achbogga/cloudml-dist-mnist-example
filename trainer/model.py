# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators.model_fn import ModeKeys as Modes

import importlib
from tensorflow.python.framework import ops
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import array_ops

import random
import numpy as np


tf.logging.set_verbosity(tf.logging.INFO)

IMAGE_HEIGHT = 28
IMAGE_WIDTH = 28
IMAGE_CHANNELS = 1
IMAGE_FLATTENED_SIZE = IMAGE_CHANNELS*IMAGE_WIDTH*IMAGE_HEIGHT
EMBEDDING_SIZE = 512
NUMBER_OF_CLASSES = 10
CENTER_LOSS_FACTOR = 0.0
CETNER_LOSS_ALFA = 0.95
WEIGHT_DECAY = 5e-4
EPOCH_SIZE = 60000

# Inception-Resnet-A
def block35(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 35x35 resnet block."""
    with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
            tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
            tower_conv2_1 = slim.conv2d(tower_conv2_0, 32, 3, scope='Conv2d_0b_3x3')
            tower_conv2_2 = slim.conv2d(tower_conv2_1, 32, 3, scope='Conv2d_0c_3x3')
        mixed = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net

# Inception-Resnet-B
def block17(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 17x17 resnet block."""
    with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 128, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 128, [1, 7],
                                        scope='Conv2d_0b_1x7')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 128, [7, 1],
                                        scope='Conv2d_0c_7x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net


# Inception-Resnet-C
def block8(net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
    """Builds the 8x8 resnet block."""
    with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
            tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
            tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
            tower_conv1_1 = slim.conv2d(tower_conv1_0, 192, [1, 3],
                                        scope='Conv2d_0b_1x3')
            tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [3, 1],
                                        scope='Conv2d_0c_3x1')
        mixed = tf.concat([tower_conv, tower_conv1_2], 3)
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        net += scale * up
        if activation_fn:
            net = activation_fn(net)
    return net
  
def reduction_a(net, k, l, m, n):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, n, 3, stride=2, padding='VALID',
                                 scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1_0 = slim.conv2d(net, k, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1_0, l, 3,
                                    scope='Conv2d_0b_3x3')
        tower_conv1_2 = slim.conv2d(tower_conv1_1, m, 3,
                                    stride=2, padding='VALID',
                                    scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
    return net

def reduction_b(net):
    with tf.variable_scope('Branch_0'):
        tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                   padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_1'):
        tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv1_1 = slim.conv2d(tower_conv1, 256, 3, stride=2,
                                    padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_2'):
        tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
        tower_conv2_1 = slim.conv2d(tower_conv2, 256, 3,
                                    scope='Conv2d_0b_3x3')
        tower_conv2_2 = slim.conv2d(tower_conv2_1, 256, 3, stride=2,
                                    padding='VALID', scope='Conv2d_1a_3x3')
    with tf.variable_scope('Branch_3'):
        tower_pool = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
    net = tf.concat([tower_conv_1, tower_conv1_1,
                        tower_conv2_2, tower_pool], 3)
    return net
  
def inference(images, keep_probability, phase_train=True, 
              bottleneck_layer_size=128, weight_decay=0.0, reuse=None):
    batch_norm_params = {
        # Decay for the moving averages.
        'decay': 0.995,
        # epsilon to prevent 0s in variance.
        'epsilon': 0.001,
        # force in-place updates of mean and variance estimates
        'updates_collections': None,
        # Moving averages ends up in the trainable variables collection
        'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    }
    
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        weights_initializer=slim.initializers.xavier_initializer(), 
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        return inception_resnet_v1(images, is_training=phase_train,
              dropout_keep_prob=keep_probability, bottleneck_layer_size=bottleneck_layer_size, reuse=reuse)


def inception_resnet_v1(inputs, is_training=True,
                        dropout_keep_prob=0.8,
                        bottleneck_layer_size=128,
                        reuse=None, 
                        scope='InceptionResnetV1'):
    """Creates the Inception Resnet V1 model.
    Args:
      inputs: a 4-D tensor of size [batch_size, height, width, 3].
      num_classes: number of predicted classes.
      is_training: whether is training or not.
      dropout_keep_prob: float, the fraction to keep before final layer.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      scope: Optional variable_scope.
    Returns:
      logits: the logits outputs of the model.
      end_points: the set of end_points from the inception model.
    """
    end_points = {}
  
    with tf.variable_scope(scope, 'InceptionResnetV1', [inputs], reuse=reuse):
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=is_training):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                                stride=1, padding='SAME'):
      
                # 149 x 149 x 32
                net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID',
                                  scope='Conv2d_1a_3x3')
                end_points['Conv2d_1a_3x3'] = net
                # 147 x 147 x 32
                net = slim.conv2d(net, 32, 3, padding='VALID',
                                  scope='Conv2d_2a_3x3')
                end_points['Conv2d_2a_3x3'] = net
                # 147 x 147 x 64
                net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
                end_points['Conv2d_2b_3x3'] = net
                # 73 x 73 x 64
                net = slim.max_pool2d(net, 3, stride=2, padding='VALID',
                                      scope='MaxPool_3a_3x3')
                end_points['MaxPool_3a_3x3'] = net
                # 73 x 73 x 80
                net = slim.conv2d(net, 80, 1, padding='VALID',
                                  scope='Conv2d_3b_1x1')
                end_points['Conv2d_3b_1x1'] = net
                # 71 x 71 x 192
                net = slim.conv2d(net, 192, 3, padding='VALID',
                                  scope='Conv2d_4a_3x3')
                end_points['Conv2d_4a_3x3'] = net
                # 35 x 35 x 256
                net = slim.conv2d(net, 256, 3, stride=2, padding='VALID',
                                  scope='Conv2d_4b_3x3')
                end_points['Conv2d_4b_3x3'] = net
                
                # 5 x Inception-resnet-A
                net = slim.repeat(net, 5, block35, scale=0.17)
                end_points['Mixed_5a'] = net
        
                # Reduction-A
                with tf.variable_scope('Mixed_6a'):
                    net = reduction_a(net, 192, 192, 256, 384)
                end_points['Mixed_6a'] = net
                
                # 10 x Inception-Resnet-B
                net = slim.repeat(net, 10, block17, scale=0.10)
                end_points['Mixed_6b'] = net
                
                # Reduction-B
                with tf.variable_scope('Mixed_7a'):
                    net = reduction_b(net)
                end_points['Mixed_7a'] = net
                
                # 5 x Inception-Resnet-C
                net = slim.repeat(net, 5, block8, scale=0.20)
                end_points['Mixed_8a'] = net
                
                net = block8(net, activation_fn=None)
                end_points['Mixed_8b'] = net
                
                with tf.variable_scope('Logits'):
                    end_points['PrePool'] = net
                    #pylint: disable=no-member
                    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                                          scope='AvgPool_1a_8x8')
                    net = slim.flatten(net)
          
                    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                                       scope='Dropout')
          
                    end_points['PreLogitsFlatten'] = net
                
                net = slim.fully_connected(net, bottleneck_layer_size, activation_fn=None, 
                        scope='Bottleneck', reuse=False)
  
    return net, end_points

def adjust_image(data):
        # Reshape to [batch, height, width, channels].
        imgs = tf.reshape(data, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
        # Adjust image size to Inception-v3 input.
        imgs = tf.image.resize_images(imgs, (256, 256))
        # Convert to RGB image.
        imgs = tf.image.grayscale_to_rgb(imgs)
        return imgs

def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
            serialized_example,
            features={
                    'image_raw': tf.FixedLenFeature([], tf.string),
                    'label': tf.FixedLenFeature([], tf.int64),
            })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image.set_shape([IMAGE_FLATTENED_SIZE])
    image = tf.cast(image, tf.float32) * (1. / 255)
    label = tf.cast(features['label'], tf.int32)

    return image, label


def input_fn(filename, batch_size=100):
    filename_queue = tf.train.string_input_producer([filename])

    image, label = read_and_decode(filename_queue)
    images, labels = tf.train.batch(
            [image, label], batch_size=batch_size,
            capacity=1000 + 3 * batch_size)

    return {'inputs': images}, labels


def get_input_fn(filename, batch_size=100):
    return lambda: input_fn(filename, batch_size)

def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    with tf.control_dependencies([centers]):
        loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers



def _cnn_model_fn(features, labels, mode):
    # Input Layer
    input_layer = tf.reshape(features['inputs'], [-1, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS])

    # Convolutional Layer #1

    conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
            inputs=dense, rate=0.4, training=(mode == Modes.TRAIN))

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    # Define operations
    if mode in (Modes.INFER, Modes.EVAL):
        predicted_indices = tf.argmax(input=logits, axis=1)
        probabilities = tf.nn.softmax(logits, name='softmax_tensor')

    if mode in (Modes.TRAIN, Modes.EVAL):
        global_step = tf.contrib.framework.get_or_create_global_step()
        label_indices = tf.cast(labels, tf.int32)
        loss = tf.losses.softmax_cross_entropy(
                onehot_labels=tf.one_hot(label_indices, depth=10), logits=logits)
        tf.summary.scalar('OptimizeLoss', loss)

    if mode == Modes.INFER:
        predictions = {
                'classes': predicted_indices,
                'probabilities': probabilities
        }
        export_outputs = {
                'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(
                mode, predictions=predictions, export_outputs=export_outputs)

    if mode == Modes.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == Modes.EVAL:
        eval_metric_ops = {
                'accuracy': tf.metrics.accuracy(label_indices, predicted_indices)
        }
        return tf.estimator.EstimatorSpec(
                mode, loss=loss, eval_metric_ops=eval_metric_ops)

def _inception_resnet_v1_model_fn(features, labels, mode):
    # Input Layer
    input_layer = adjust_image(features['inputs'])

    # Create a queue that produces indices into the image_list and label_list 
    labels = ops.convert_to_tensor(labels, dtype=tf.int32)
    range_size = array_ops.shape(labels)[0]
    learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
            
    phase_train_placeholder = tf.placeholder_with_default(True, shape=(), name='phase_train')
    labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels')
    # Build the inference graph
    prelogits, _ = inference(input_layer, 0.4, phase_train=phase_train_placeholder, bottleneck_layer_size=EMBEDDING_SIZE, weight_decay=5e-4)
    logits = slim.fully_connected(prelogits, NUMBER_OF_CLASSES, activation_fn=None, 
                    weights_initializer=slim.initializers.xavier_initializer(WEIGHT_DECAY), 
                    weights_regularizer=slim.l2_regularizer(WEIGHT_DECAY),
                    scope='Logits', reuse=False)

    embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')

    # Norm for the prelogits
    eps = 1e-4
    prelogits_norm = tf.reduce_mean(tf.norm(tf.abs(prelogits)+eps, ord=1.0, axis=1))
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_norm * 0.0)

    # Add center loss
    prelogits_center_loss, _ = center_loss(prelogits, labels, CETNER_LOSS_ALFA, NUMBER_OF_CLASSES)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * CENTER_LOSS_FACTOR)

    #learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,1*EPOCH_SIZE, 0.1, staircase=True)
    #tf.summary.scalar('learning_rate', learning_rate)

    # Calculate the average cross entropy loss across the batch
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)
    
    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.cast(labels, tf.int64)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)
    
    # Calculate the total losses
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n([cross_entropy_mean] + regularization_losses, name='total_loss')
    # Define operations
    if mode in (Modes.INFER, Modes.EVAL):
        predicted_indices = tf.argmax(input=logits, axis=1)
        probabilities = tf.nn.softmax(logits, name='softmax_tensor')

    if mode in (Modes.TRAIN, Modes.EVAL):
        global_step = tf.contrib.framework.get_or_create_global_step()
        label_indices = tf.cast(labels, tf.int32)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(label_indices, depth=10), logits=logits)
        tf.summary.scalar('OptimizeLoss', loss)

    if mode == Modes.INFER:
        predictions = {
                'classes': predicted_indices,
                'probabilities': probabilities
        }
        export_outputs = {
                'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)

    if mode == Modes.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    if mode == Modes.EVAL:
        eval_metric_ops = {'accuracy': tf.metrics.accuracy(label_indices, predicted_indices)}
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)



def build_estimator(model_dir):
    return tf.estimator.Estimator(
            model_fn=_inception_resnet_v1_model_fn,
            model_dir=model_dir,
            config=tf.contrib.learn.RunConfig(save_checkpoints_secs=180))


def serving_input_fn():
    inputs = {'inputs': tf.placeholder(tf.float32, [None, IMAGE_FLATTENED_SIZE])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

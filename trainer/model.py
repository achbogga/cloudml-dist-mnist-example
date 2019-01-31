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
import facenet
from tensorflow.python.framework import ops
import tensorflow.contrib.slim as slim

import argparse
import lfw_custom_data as lfw
import h5py
import math
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import array_ops
from datetime import datetime
import os.path
import time
import sys
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

    network = importlib.import_module('models.inception_resnet_v1')
    # Create a queue that produces indices into the image_list and label_list 
    labels = ops.convert_to_tensor(labels, dtype=tf.int32)
    range_size = array_ops.shape(labels)[0]
    learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')
        
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
    labels_placeholder = tf.placeholder(tf.int32, shape=(None,1), name='labels')
    # Build the inference graph
    prelogits, _ = network.inference(input_layer, 0.4, phase_train=phase_train_placeholder, bottleneck_layer_size=EMBEDDING_SIZE, weight_decay=5e-4)
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
    prelogits_center_loss, _ = facenet.center_loss(prelogits, labels, CETNER_LOSS_ALFA, nrof_classes)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, prelogits_center_loss * CENTER_LOSS_FACTOR)

    learning_rate = tf.train.exponential_decay(learning_rate_placeholder, global_step,1*EPOCH_SIZE, 0.1, staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

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
		eval_metric_ops = {
		    'accuracy': tf.metrics.accuracy(label_indices, predicted_indices)
		}
		return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)



def build_estimator(model_dir):
  return tf.estimator.Estimator(
      model_fn=_inception_resnet_v1_model_fn,
      model_dir=model_dir,
      config=tf.contrib.learn.RunConfig(save_checkpoints_secs=180))


def serving_input_fn():
  inputs = {'inputs': tf.placeholder(tf.float32, [None, IMAGE_FLATTENED_SIZE])}
  return tf.estimator.export.ServingInputReceiver(inputs, inputs)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

# NUM_CLASSES = 10

IMAGE_SIZE = 32
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_IMAGES = 3
TOTAL_PIXELS = IMAGE_PIXELS * NUM_IMAGES

images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                       TOTAL_PIXELS))
labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))

def inference(images, hidden1_units, hidden2_units):
  """Build the MNIST model up to where it may be used for inference.
  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.
  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([TOTAL_PIXELS, hidden1_units],
                            stddev=1.0 / math.sqrt(float(TOTAL_PIXELSf))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(hidden2, weights) + biases
  return logits
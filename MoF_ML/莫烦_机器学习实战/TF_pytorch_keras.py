import tensorflow as tf
import torch
import os

# 1/2+1/2^2+1/2^3+...+1/2^50

# # python
# x = 0
# y = 1
# for iteration in range(50):
#     x = x + y
#     y = y / 2
# print(x)

import tensorflow as tf

# tf.enable_eager_execution()
print(tf.__version__)

x = tf.constant(0.)
y = tf.constant(1.)
for iteration in range(50):
    x = x + y
    y = y / 2
print(x.numpy())





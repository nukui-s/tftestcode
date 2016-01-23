import numpy as np
import os
os.system("rm -rf test")

import tensorflow as tf

x = tf.Variable([1.,2.], name="x")
diag_x = tf.diag(x)
loss = tf.reduce_sum(diag_x)
opt = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    sess.run(opt)

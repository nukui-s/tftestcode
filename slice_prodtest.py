import numpy as np
import tensorflow as tf


mat1 = np.random.rand(10,2)
mat2 = np.ones([10,1],dtype=np.float32)

var1, var2 = tf.train.slice_input_producer([mat1, mat2])

sess = tf.InteractiveSession()

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

step = 0
while not coord.should_stop():
    print(sess.run(var1))
    print(sess.run(var2))
    step += 1
    if step > 2: coord.request_stop()
coord.join(threads)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

step = 0
while not coord.should_stop():
    print(sess.run(var1))
    print(sess.run(var2))
    step += 1
    if step > 2: coord.request_stop()
coord.join(threads)

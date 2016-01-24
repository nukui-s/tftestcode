import numpy as np
import tensorflow as tf


class ResultHolder(object):

    def __init__(self):
        self.acc = 0

    def myloop(self, coord):
        i = 0
        while not coord.should_stop():
            print(self.acc)
            if self.acc > 50:
                break
            z = self.sess.run(self.z)
            self.acc += z
            i += 1
        coord.request_stop()

    def inference(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            indices = tf.constant([[1.],[2.],[3.]])
            values = tf.constant([[3.],[4.],[7.]])
            ind, val = tf.train.slice_input_producer([indices, values],capacity=30)
            self.queue = tf.FIFOQueue(10000, ["float","float"],
                                    shapes=[[1],[1]])
            x, y = self.queue.dequeue()
            self.z = ind + val
            self.sess = tf.Session()

    def enqueue(self, x, y):
        with self.graph.as_default():
            enq = self.queue.enqueue_many([x,y])
        self.sess.run(enq)

    def get_z(self):
        return self.sess.run(self.z)

x = np.array([1,2,3,4]).astype(np.float32).reshape(4,1)
y = np.array([6,3,5,6]).astype(np.float32).reshape(4,1)
myclass = ResultHolder()
myclass.inference()
quit()
#for i in range(100): myclass.enqueue(x,y)
coord = tf.train.Coordinator()

method = myclass.myloop
threads = [tf.train.threading.Thread(target=method, args=(coord,))
            for i in range(4)]

for t in threads: t.start()
coord.join(threads)

myclass.acc = 0

print("Twice")
coord = tf.train.Coordinator()
threads = [tf.train.threading.Thread(target=method, args=(coord,))
            for i in range(4)]
for t in threads: t.start()
coord.join(threads)

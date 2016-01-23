import numpy as np
import tensorflow as tf




def inference(sess):
    graph = sess.graph
    with graph.as_default():
        queue = tf.FIFOQueue(100, ["float","float"])
        x, y = queue.dequeue()
        z = x + y
    return queue, z


class ResultHolder(object):

    def __init__(self):
        self.acc = 0

    def myloop(self, coord):
        i = 0
        while not coord.should_stop():
            print(self.acc)
            if self.acc > 19:
                break
            z = self.sess.run(self.z)
            print(z,i)
            self.acc += z
            i += 1
        coord.request_stop()

    def inference(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.queue = tf.FIFOQueue(100, ["float","float"],
                                    shapes=[[1],[1]])
            x, y = self.queue.dequeue()
            self.z = x + y
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
for i in range(10): myclass.enqueue(x,y)
coord = tf.train.Coordinator()

method = myclass.myloop
threads = [tf.train.threading.Thread(target=method, args=(coord,))
            for i in range(4)]

for t in threads: t.start()
coord.join(threads)

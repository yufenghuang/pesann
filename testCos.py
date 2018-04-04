# -*- coding: utf-8 -*-

import tensorflow as tf
import tf_func as tff
import numpy as np
import matplotlib.pyplot as plt

Rc = 6.2

tfX = tf.placeholder(tf.float32, shape=(None))

tfY = tff.tf_getCos2(tfX, 100)

tfdY = tff.tf_getdCos2(tfX, 100)

sess = tf.Session()

x = np.linspace(-1.1, -0.7, 1000)

y = sess.run(tfY, feed_dict={tfX: x})

dy = sess.run(tfdY, feed_dict={tfX: x})

plt.figure()
plt.plot(Rc*(x+2)/4, y[:,:10])
#plt.plot(x, dy)
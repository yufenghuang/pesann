# -*- coding: utf-8 -*-

import tensorflow as tf
import tf_func as tff
import numpy as np
import matplotlib.pyplot as plt

tfX = tf.placeholder(tf.float32, shape=(None))

tfY = tff.tf_getCos2(tfX, 8)

tfdY = tff.tf_getdCos2(tfX, 8)

sess = tf.Session()

x = np.linspace(-2, 2, 1000)

y = sess.run(tfY, feed_dict={tfX: x})

dy = sess.run(tfdY, feed_dict={tfX: x})

plt.figure()
plt.plot(x, y)
plt.plot(x, dy)
import tensorflow as tf
indicies = [1, 2, 0, 1, 2, 3, 4]
depth = 5
one_hot= tf.one_hot(indicies, depth)

with tf.Session() as sess:
    print(sess.run(one_hot))
import tensorflow as tf
import zhusuan as zs
from keras.datasets import mnist
from PIL import Image
import numpy as np
import keras
import time


@zs.reuse('model')
def vae(observed, x_dim, z_dim, n, y):
    with zs.BayesianNet(observed=observed) as model:
        z_mean = tf.zeros([n, z_dim])
        z = zs.Normal('z', z_mean, std=1., group_ndims=1)
        z_plus_y = tf.concat([z, tf.cast(y, tf.float32)], axis=1)
        lx_z = tf.layers.dense(z_plus_y, 500, activation=tf.nn.relu)
        lx_z = tf.layers.dense(lx_z, 500, activation=tf.nn.relu)
        x_logits = tf.layers.dense(lx_z, x_dim)
        x_mean = zs.Implicit("x_mean", tf.sigmoid(x_logits), group_ndims=1)
        x = zs.Bernoulli('x', x_logits, group_ndims=1)
    return model, x_mean


@zs.reuse('variational')
def q_net(x, y, z_dim):
    with zs.BayesianNet() as variational:
        x_plus_y = tf.concat([x, y], axis=1)
        lz_x = tf.layers.dense(tf.to_float(x_plus_y), 500, activation=tf.nn.relu)
        lz_x = tf.layers.dense(lz_x, 500, activation=tf.nn.relu)
        z_mean = tf.layers.dense(lz_x, z_dim)
        z_logstd = tf.layers.dense(lz_x, z_dim)
        z = zs.Normal('z', z_mean, logstd=z_logstd, group_ndims=1)
    return variational


def save_figs(img, name):
    img = np.reshape(np.asarray(img), (10, 10, 28, 28))
    imgs = np.zeros((10 * 28, 10 * 28), dtype=np.float)
    for i in range(10):
        for j in range(10):
            imgs[i * 28: (i + 1) * 28, j * 28: (j + 1) * 28] = img[i, j, :, :]
    im = Image.fromarray(np.uint8(imgs * 255))
    im.save(name)


if __name__ == '__main__':
    # Load MNIST
    IMAGE_SIZE = 28*28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.reshape(x_train, [-1, IMAGE_SIZE])
    x_test = np.reshape(x_test, [-1, IMAGE_SIZE])
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    idx = {}
    for i in range(10):
        idx[i] = np.where(y_test == i)[0]

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    # Define model parameters
    x_dim = x_train.shape[1]
    y_dim = y_train.shape[1]
    z_dim = 40

    # Build the computation graph
    x_input = tf.placeholder(tf.float32, shape=[None, x_dim], name='x')
    y_input = tf.placeholder(tf.int32, shape=[None, y_dim], name='y')
    x = tf.to_int32(tf.less(tf.random_uniform(tf.shape(x_input)), x_input))
    n = tf.shape(x)[0]

    x_reconstruct = None
    def log_joint(observed):
        global x_reconstruct
        model, x_reconstruct = vae(observed, x_dim, z_dim, n, y_input)
        log_pz, log_px_z = model.local_log_prob(['z', 'x'])
        return log_pz + log_px_z

    variational = q_net(x, y_input, z_dim)
    qz_samples, log_qz = variational.query('z', outputs=True,
                                           local_log_prob=True)
    lower_bound = zs.variational.elbo(log_joint,
                                      observed={'x': x},
                                      latent={'z': [qz_samples, log_qz]},
                                      axis=0)
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    infer_op = optimizer.minimize(cost)

    # Generate images
    n_gen = 100
    _, x_mean = vae({}, x_dim, z_dim, n_gen, y_input)
    # x_gen = tf.reshape(x_mean, [-1, 28, 28, 1])
    x_rec = tf.reshape(x_reconstruct, [-1, 28, 28, 1])

    # Define training/evaluation parameters
    epochs = 3000
    batch_size = 256
    iters = x_train.shape[0] // batch_size
    save_freq = 10

    # Run the inference
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, epochs + 1):
            time_epoch = -time.time()
            np.random.shuffle(x_train)
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                _, lb = sess.run([infer_op, lower_bound],
                                 feed_dict={x_input: x_batch,
                                            y_input: y_batch})
                lbs.append(lb)
            time_epoch += time.time()
            print('Epoch {} ({:.1f}s): Lower bound = {}'.format(
                epoch, time_epoch, np.mean(lbs)))

            if epoch % save_freq == 0:
                # y_batch = keras.utils.to_categorical(np.tile(np.arange(10), reps=10), num_classes=10)
                # img = sess.run(x_gen, feed_dict={y_input: y_batch})
                # save_figs(img, 'figs/cvae.epoch.{}.png'.format(epoch))

                test_idx = []
                for i in range(10):
                    test_idx.append(idx[i][np.random.randint(0, len(idx), 10)])
                test_batch_idx = np.stack(test_idx, axis=1).flatten()

                img = sess.run(x_reconstruct, feed_dict={x_input: x_test[test_batch_idx],
                                                        y_input: y_test[test_batch_idx]})
                save_figs(img, 'figs-v1/cvae.reconstruct.epoch.{}.png'.format(epoch))

from __future__ import division
import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use('Agg')
def generator_conditional(z, conditioning):  # Convolution Generator
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        z_combine = tf.concat([z, conditioning], -1)
        conv1_g = tf.layers.conv1d(inputs=z_combine, filters=256, kernel_size=5, padding='same')
        # conv1_g_bn = tf.layers.batch_normalization(conv1_g, training=training)
        conv1_g = tf.nn.leaky_relu(conv1_g)
        conv2_g = tf.layers.conv1d(inputs=conv1_g, filters=128, kernel_size=3, padding='same')
        conv2_g = tf.nn.leaky_relu(conv2_g)
        conv3_g = tf.layers.conv1d(inputs=conv2_g, filters=64, kernel_size=3, padding='same')
        conv3_g = tf.nn.leaky_relu(conv3_g)
        conv4_g = tf.layers.conv1d(inputs=conv3_g, filters=2, kernel_size=3, padding='same')
        return conv4_g


def discriminator_condintional(x, conditioning):
    with tf.variable_scope("discriminator", reuse=tf.AUTO_REUSE):
        z_combine = tf.concat([x, conditioning], -1)
        conv1 = tf.layers.conv1d(inputs=z_combine, filters=256, kernel_size=5, padding='same')
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.reduce_mean(conv1, axis=0, keep_dims=True)
        conv2 = tf.layers.conv1d(inputs=conv1, filters=128, kernel_size=3, padding='same')
        conv2 = tf.nn.relu(conv2)
        conv3 = tf.layers.conv1d(inputs=conv2, filters=64, kernel_size=3, padding='same')
        conv3 = tf.nn.relu(conv3)
        conv4 = tf.layers.conv1d(inputs=conv3, filters=16, kernel_size=3, padding='same')
        FC = tf.nn.relu(tf.layers.dense(conv4, 100, activation=None))
        D_logit = tf.layers.dense(FC, 1, activation=None)
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob, D_logit


def encoding(x):
    with tf.variable_scope("encoding", reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv1d(inputs=x, filters=256, kernel_size=5, padding='same')
        conv1 = tf.nn.relu(conv1)
        conv2 = tf.layers.conv1d(inputs=conv1, filters=128, kernel_size=3, padding='same')
        conv2 = tf.nn.relu(conv2)
        conv3 = tf.layers.conv1d(inputs=conv2, filters=64, kernel_size=3, padding='same')
        conv3 = tf.nn.relu(conv3)
        conv4 = tf.layers.conv1d(inputs=conv3, filters=2, kernel_size=3, padding='same')
        layer_4_normalized = tf.scalar_mul(tf.sqrt(tf.cast(block_length, tf.float32)),
                                           tf.nn.l2_normalize(conv4, dim=1))  # normalize the encoding.
        return layer_4_normalized


def decoding(x):
    with tf.variable_scope("decoding", reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv1d(inputs=x, filters=256, kernel_size=5, padding='same')
        conv1 = tf.nn.relu(conv1)
        conv2_ori = tf.layers.conv1d(inputs=conv1, filters=128, kernel_size=5, padding='same')
        conv2 = tf.nn.relu(conv2_ori)
        conv2 = tf.layers.conv1d(inputs=conv2, filters=128, kernel_size=5, padding='same')
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.layers.conv1d(inputs=conv2, filters=128, kernel_size=5, padding='same')
        conv2 += conv2_ori
        conv2 = tf.nn.relu(conv2)
        conv3_ori = tf.layers.conv1d(inputs=conv2, filters=64, kernel_size=5, padding='same')
        conv3 = tf.nn.relu(conv3_ori)
        conv3 = tf.layers.conv1d(inputs=conv3, filters=64, kernel_size=5, padding='same')
        conv3 = tf.nn.relu(conv3)
        conv3 = tf.layers.conv1d(inputs=conv3, filters=64, kernel_size=3, padding='same')
        conv3 += conv3_ori
        conv3 = tf.nn.relu(conv3)
        conv4 = tf.layers.conv1d(inputs=conv3, filters=32, kernel_size=3, padding='same')
        conv4 = tf.nn.relu(conv4)
        Decoding_logit = tf.layers.conv1d(inputs=conv4, filters=1, kernel_size=3, padding='same')
        Decoding_prob = tf.nn.sigmoid(Decoding_logit)

        return Decoding_logit, Decoding_prob


def sample_Z(sample_size):
    ''' Sampling the generation noise Z from normal distribution'''
    return np.random.normal(size=sample_size)


def sample_uniformly(sample_size):
    return np.random.randint(size=sample_size, low=-15, high=15) / 10


def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


def generate_batch_data(batch_size):
    global start_idx, data
    if start_idx + batch_size >= data_size:
        start_idx = 0
        data = np.random.binomial(1, 0.5, [data_size, block_length, 1])
    batch_x = data[start_idx:start_idx + batch_size]
    start_idx += batch_size
    return batch_x


""" Start of the Main function """

''' Building the Graph'''
batch_size = 512
block_length = 128
condition_depth = 2
Z_dim_c = 16
learning_rate = 1e-4

X = tf.placeholder(tf.float32, shape=[None, block_length, 1])
E = encoding(X)
Z = tf.placeholder(tf.float32, shape=[None, block_length, Z_dim_c])
Noise_std = tf.placeholder(tf.float32, shape=[])

G_sample = generator_conditional(Z, E)
R_sample = gaussian_noise_layer(E, Noise_std)
print("shapes G and R,", G_sample, R_sample)
R_decodings_logit, R_decodings_prob = decoding(R_sample)
G_decodings_logit, G_decodings_prob = decoding(G_sample)

encodings_uniform_generated = tf.placeholder(tf.float32, shape=[None, block_length, 2])
G_sample_uniform = generator_conditional(Z, encodings_uniform_generated)
R_sample_uniform = gaussian_noise_layer(encodings_uniform_generated, Noise_std)
D_prob_real, D_logit_real = discriminator_condintional(R_sample_uniform, encodings_uniform_generated)
D_prob_fake, D_logit_fake = discriminator_condintional(G_sample_uniform, encodings_uniform_generated)

Disc_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
Gen_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
Tx_vars = [v for v in tf.trainable_variables() if v.name.startswith('encoding')]
Rx_vars = [v for v in tf.trainable_variables() if v.name.startswith('decoding')]

D_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
D_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))

D_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(D_loss, var_list=Disc_vars)
G_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(G_loss, var_list=Gen_vars)

loss_receiver_R = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=R_decodings_logit, labels=X))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
Rx_solver = optimizer.minimize(loss_receiver_R, var_list=Rx_vars)
loss_receiver_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=G_decodings_logit, labels=X))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
Tx_solver = optimizer.minimize(loss_receiver_G, var_list=Tx_vars)
accuracy_R = tf.reduce_mean(tf.cast((tf.abs(R_decodings_prob - X) > 0.5), tf.float32))
WER_R = 1 - tf.reduce_mean(tf.cast(tf.reduce_all(tf.abs(R_decodings_prob - X) < 0.5, 1), tf.float32))
accuracy_G = tf.reduce_mean(tf.cast((tf.abs(G_decodings_prob - X) > 0.5), tf.float32))
WER_G = 1 - tf.reduce_mean(tf.cast(tf.reduce_all(tf.abs(G_decodings_prob - X) < 0.5, 1), tf.float32))

init = tf.global_variables_initializer()
number_steps_receiver = 0
number_steps_channel = 0
number_steps_transmitter = 0
display_step = 100
batch_size = 320
number_iterations = 1000  # in each iteration, the receiver, the transmitter and the channel will be updated

EbNo_train = 3
EbNo_train = 10 ** (EbNo_train / 10)

EbNo_train_GAN = 6
EbNo_train_GAN = 10 ** (EbNo_train_GAN / 10)

EbNo_test = 6
EbNo_test = 10 ** (EbNo_test / 10)

R = 0.5

N_training = int(1e6)
data = np.random.binomial(1, 0.5, [N_training, block_length, 1])
N_val = int(1e4)
val_data = np.random.binomial(1, 0.5, [N_val, block_length, 1])
N_test = int(1e6)
test_data = np.random.binomial(1, 0.5, [N_test, block_length, 1])
data_size = len(data)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
saver = tf.train.Saver()
load_pretrain_model = True

with tf.Session(config=config) as sess:
    start_idx = 0
    print('Start init')
    sess.run(tf.global_variables_initializer())
    for iteration in range(number_iterations):
        print("iteration is ", iteration)
        number_steps_transmitter += 5000
        number_steps_receiver += 5000
        number_steps_channel += 2000
        ''' =========== Training the Channel Simulator ======== '''
        for step in range(number_steps_channel):
            if step % 100 == 0:
                print("Training ChannelGAN, step is ", step)
            batch_x = generate_batch_data(int(batch_size / 2))
            encoded_data = sess.run([E], feed_dict={X: batch_x})
            random_data = sample_uniformly([int(batch_size / 2), block_length, 2])

            input_data = np.concatenate((np.asarray(encoded_data).reshape([int(batch_size / 2), block_length, 2])
                                         + np.random.normal(0, 0.1, size=([int(batch_size / 2), block_length, 2])),
                                         random_data), axis=0)

            _, D_loss_curr = sess.run([D_solver, D_loss],
                                      feed_dict={encodings_uniform_generated: input_data,
                                                 Z: sample_Z([batch_size, block_length, Z_dim_c]),
                                                 Noise_std: (np.sqrt(1 / (2 * R * EbNo_train_GAN)))})
            _, G_loss_curr = sess.run([G_solver, G_loss],
                                      feed_dict={encodings_uniform_generated: input_data,
                                                 Z: sample_Z([batch_size, block_length, Z_dim_c]),
                                                 Noise_std: (np.sqrt(1 / (2 * R * EbNo_train_GAN)))})

        ''' =========== Training the Transmitter ==== '''
        for step in range(number_steps_transmitter):
            if step % 100 == 0:
                print("Training transmitter, step is ", step)

            batch_x = generate_batch_data(batch_size)
            sess.run(Tx_solver, feed_dict={X: batch_x, Z: sample_Z([batch_size, block_length, Z_dim_c]),
                                           Noise_std: (np.sqrt(1 / (2 * R * EbNo_train)))
                                           })

        ''' ========== Training the Receiver ============== '''

        for step in range(number_steps_receiver):
            if step % 100 == 0:
                print("Training receiver, step is ", step)
            batch_x = generate_batch_data(batch_size)
            sess.run(Rx_solver, feed_dict={X: batch_x, Noise_std: (np.sqrt(1 / (2 * R * EbNo_train)))})

        '''  ----- Testing ----  '''

        loss, acc = sess.run([loss_receiver_R, accuracy_R],
                             feed_dict={X: batch_x, Noise_std: np.sqrt(1 / (2 * 2 * R * EbNo_train))})
        print("Real Channel Evaluation:", "Step " + str(step) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss) + ", Training Accuracy= " + \
              "{:.3f}".format(acc))

        loss, acc = sess.run([loss_receiver_G, accuracy_G],
                             feed_dict={X: batch_x, Z: sample_Z([batch_size, block_length, Z_dim_c]),
                                        Noise_std: np.sqrt(1 / (2 * 2 * R * EbNo_train))
                                        })
        print("Generated Channel Evaluation:", "Step " + str(step) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss) + ", Training Accuracy= " + \
              "{:.3f}".format(acc))

        EbNodB_range = np.arange(0, 8.5, 0.5)
        ber = np.ones(len(EbNodB_range))
        for n in range(0, len(EbNodB_range)):
            EbNo = 10.0 ** (EbNodB_range[n] / 10.0)
            ber[n] = 1 - sess.run(accuracy_R,
                                  feed_dict={X: test_data, Noise_std: (np.sqrt(1 / (2 * R * EbNo)))})
            print('SNR:', EbNodB_range[n], 'BER:', ber[n])




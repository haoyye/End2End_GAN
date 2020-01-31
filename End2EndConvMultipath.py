from __future__ import division
import numpy as np
import tensorflow as tf
''' This file aims to solve the end to end communication problem in Rayleigh fading channel '''
''' The condition of channel GAN is the encoding and information h '''
''' We should compare with baseline that equalizor of Rayleigh fading'''
CP = 16
N = 64
def generator_conditional(z, conditioning):  # Convolution Generator
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        z_combine = tf.concat([z, conditioning], -1)
        conv1_g = tf.layers.conv1d(inputs=z_combine, filters=256, kernel_size=5, padding='same')
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
        layer_4_normalized = tf.scalar_mul(tf.sqrt(tf.cast((block_length / 2), tf.float32)),
                                           tf.nn.l2_normalize(conv4, dim=1))  # normalize the encoding.
        return layer_4_normalized
def decoding(x, channel_info):
    x_combine = tf.concat([x, channel_info], -1)
    with tf.variable_scope("decoding", reuse=tf.AUTO_REUSE):
        conv1 = tf.layers.conv1d(inputs=x_combine, filters=256, kernel_size=5, padding='same')
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
        return Decoding_logit[:, 0:block_length], Decoding_prob[:, 0:block_length]

def sample_Z(sample_size):
    ''' Sampling the generation noise Z from normal distribution '''
    return np.random.normal(size=sample_size)

def sample_uniformly(sample_size):
    return np.random.randint(size=sample_size, low=-15, high=15) / 10

def Multipath_layer(x, h_r, h_i, std):
    x_pad = tf.pad(x, tf.constant([[0, 0], [0, L], [0, 0]]))
    h_r = tf.reshape(h_r, [-1, L, 1])
    h_i = tf.reshape(h_i, [-1, L, 1])
    x_r = tf.reshape(x_pad[:, :, 0], [-1, block_length + L, 1])
    x_i = tf.reshape(x_pad[:, :, 1], [-1, block_length + L, 1])
    def convolution(x, h):
        y = x * tf.reshape(h[:, 0, 0], [-1, 1, 1])
        for i in range(1, L):
            cur = x * tf.reshape(h[:, i, 0], [-1, 1, 1])
            cur = tf.concat([cur[:, -i:, :], cur[:, :-i, :]], 1)
            y += cur
        return y
    o_r = convolution(x_r, h_r) - convolution(x_i, h_i)
    o_i = convolution(x_r, h_i) + convolution(x_i, h_r)
    output = tf.concat([o_r, o_i], -1)
    noise = tf.random_normal(shape=tf.shape(output), mean=0.0, stddev=std, dtype=tf.float32)
    output += noise
    return output


def generate_channel(PDP):
    """ Generate channel based on the PDP """
    h = 1 / np.sqrt(2) * (np.random.normal(size=len(PDP)) + 1j * np.random.normal(size=len(PDP))) * np.sqrt(PDP)
    return h


def generate_channel_parts(PDP, sample_size):
    """ Generate real and imagary part of channel """
    h = 1 / np.sqrt(2) * np.sqrt(PDP) * np.random.normal(size=sample_size)
    return h


def generate_PDP(L):
    """ Generate the PDP for channel generation """
    PDP = np.ones(L)
    PDP = PDP / sum(PDP)
    return PDP


def sample_h(sample_size):
    """ sampling the h """
    return np.random.normal(size=sample_size) / np.sqrt(2.)


def encoding_padding(encoding):
    """ padding the encodings s.t. the output number will be the same as the conv """
    paddings = tf.constant([[0, 0], [0, L], [0, 0]])
    encoding_padding = tf.pad(encoding, paddings, "CONSTANT")
    return encoding_padding


""" Start of the Main function """
''' Building the Graph'''
batch_size = 512
block_length = 64
condition_depth = 4
Z_dim_c = 16
learning_rate = 1e-4
L = 3
condition_length = block_length + L
channel_PDP = generate_PDP(L)

X = tf.placeholder(tf.float32, shape=[None, block_length, 1])
E_r = encoding(X)

E = encoding_padding(E_r)
Z = tf.placeholder(tf.float32, shape=[None, condition_length, Z_dim_c])
Noise_std = tf.placeholder(tf.float32, shape=[])

h_r = tf.placeholder(tf.float32, shape=[None, L])
h_i = tf.placeholder(tf.float32, shape=[None, L])

h_r_noise = tf.add(h_r, tf.random_normal(shape=tf.shape(h_i), mean=0.0, stddev=Noise_std / 8, dtype=tf.float32))
h_i_noise = tf.add(h_i, tf.random_normal(shape=tf.shape(h_r), mean=0.0, stddev=Noise_std / 8, dtype=tf.float32))

Channel_info = tf.tile(tf.concat([tf.reshape(h_r, [-1, 1, L]), tf.reshape(h_i, [-1, 1, L])], -1),
                       [1, condition_length, 1])

Conditions = tf.concat([E, Channel_info], axis=-1)
G_sample = generator_conditional(Z, Conditions)
print("shape of E", E, h_r, h_i)
R_sample = Multipath_layer(E_r, h_r, h_i, Noise_std)
R_decodings_logit, R_decodings_prob = decoding(R_sample, Channel_info)
G_decodings_logit, G_decodings_prob = decoding(G_sample, Channel_info)

encodings_uniform_generated = tf.placeholder(tf.float32, shape=[None, block_length, 2])
encodings_uniform_generated_padding = encoding_padding(encodings_uniform_generated)
Conditions_uniform = tf.concat([encodings_uniform_generated_padding, Channel_info], axis=-1)

print("shapes G and R and channel info", G_sample, R_sample, encodings_uniform_generated_padding)
G_sample_uniform = generator_conditional(Z, Conditions_uniform)
R_sample_uniform = Multipath_layer(encodings_uniform_generated, h_r, h_i, Noise_std)
D_prob_real, D_logit_real = discriminator_condintional(R_sample_uniform, Conditions_uniform)
D_prob_fake, D_logit_fake = discriminator_condintional(G_sample_uniform, Conditions_uniform)

Disc_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
Gen_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]
Tx_vars = [v for v in tf.trainable_variables() if v.name.startswith('encoding')]
Rx_vars = [v for v in tf.trainable_variables() if v.name.startswith('decoding')]

''' Standard GAN '''
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
batch_size = 1000
number_iterations = 1000  # in each iteration, the receiver, the transmitter and the channel will be updated

EbNo_train = 20.
EbNo_train = 10. ** (EbNo_train / 10.)

EbNo_train_GAN = 35
EbNo_train_GAN = 10. ** (EbNo_train_GAN / 10.)

EbNo_test = 15.
EbNo_test = 10. ** (EbNo_test / 10.)

R = 0.5

N_training = int(5e6)
data = np.random.binomial(1, 0.5, [N_training, block_length, 1])
data_size = len(data)
N_val = int(1e4)
val_data = np.random.binomial(1, 0.5, [N_val, block_length, 1])
N_test = int(1e4)
test_data = np.random.binomial(1, 0.5, [N_test, block_length, 1])

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

def generate_batch_data(batch_size):
    global start_idx, data
    if start_idx + batch_size >= data_size:
        start_idx = 0
        data = np.random.binomial(1, 0.5, [data_size, block_length, 1])
    batch_x = data[start_idx:start_idx + batch_size]
    start_idx += batch_size
    return batch_x
load_pretrain_model = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    start_idx = 0
    for iteration in range(number_iterations):
        print("iteration is ", iteration)
        number_steps_transmitter += 5000
        number_steps_receiver += 5000
        number_steps_channel += 2000
        ''' =========== Training the Channel Simulator ======== '''
        for step in range(number_steps_channel):
            if step % 100 == 0:
                print("Training ChannelGAN, step is ", step)
            # batch_x = data[start_idx:start_idx + int(batch_size / 2), :]
            batch_x = generate_batch_data(int(batch_size / 2))
            encoded_data = sess.run([E_r], feed_dict={X: batch_x})
            random_data = sample_uniformly([int(batch_size / 2), block_length, 2])
            # print(np.asarray(encoded_data).shape, np.asarray(random_data).shape)
            input_data = np.concatenate((np.asarray(encoded_data).reshape([int(batch_size / 2), block_length, 2])
                                         + np.random.normal(0, 0.1, size=([int(batch_size / 2), block_length, 2])),
                                         random_data), axis=0)

            _, D_loss_curr = sess.run([D_solver, D_loss],
                                      feed_dict={encodings_uniform_generated: input_data,
                                                 h_i: generate_channel_parts(channel_PDP, [batch_size, L]),
                                                 h_r: generate_channel_parts(channel_PDP, [batch_size, L]),
                                                 Z: sample_Z([batch_size, condition_length, Z_dim_c]),
                                                 Noise_std: (np.sqrt(1 / (2 * R * EbNo_train_GAN)))})
            _, G_loss_curr = sess.run([G_solver, G_loss],
                                      feed_dict={encodings_uniform_generated: input_data,
                                                 h_i: generate_channel_parts(channel_PDP, [batch_size, L]),
                                                 h_r: generate_channel_parts(channel_PDP, [batch_size, L]),
                                                 Z: sample_Z([batch_size, condition_length, Z_dim_c]),
                                                 Noise_std: (np.sqrt(1 / (2 * R * EbNo_train_GAN)))})

        ''' =========== Training the Transmitter ==== '''
        for step in range(number_steps_transmitter):
            if step % 100 == 0:
                print("Training transmitter, step is ", step)

            # batch_x = data[start_idx:start_idx + batch_size, :]
            batch_x = generate_batch_data(batch_size)
            sess.run(Tx_solver, feed_dict={X: batch_x, Z: sample_Z([batch_size, condition_length, Z_dim_c]),
                                           h_i: generate_channel_parts(channel_PDP, [batch_size, L]),
                                           h_r: generate_channel_parts(channel_PDP, [batch_size, L]),
                                           Noise_std: (np.sqrt(1 / (2 * R * EbNo_train)))
                                           })

        ''' ========== Training the Receiver ============== '''

        for step in range(number_steps_receiver):
            if step % 100 == 0:
                print("Training receiver, step is ", step)
            batch_x = generate_batch_data(batch_size)
            sess.run(Rx_solver, feed_dict={X: batch_x,
                                           h_i: generate_channel_parts(channel_PDP, [batch_size, L]),
                                           h_r: generate_channel_parts(channel_PDP, [batch_size, L]),
                                           Noise_std: (np.sqrt(1 / (2 * R * EbNo_train)))})

        '''  ----- Testing ----  '''

        loss, acc = sess.run([loss_receiver_R, accuracy_R],
                             feed_dict={X: batch_x,
                                        h_i: generate_channel_parts(channel_PDP, [batch_size, L]),
                                        h_r: generate_channel_parts(channel_PDP, [batch_size, L]),
                                        Noise_std: np.sqrt(1 / (2 * R * EbNo_train))})
        print("Real Channel Evaluation:", "Step " + str(step) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss) + ", Training Accuracy= " + \
              "{:.3f}".format(acc))

        loss, acc = sess.run([loss_receiver_G, accuracy_G],
                             feed_dict={X: batch_x,
                                        h_i: generate_channel_parts(channel_PDP, [batch_size, L]),
                                        h_r: generate_channel_parts(channel_PDP, [batch_size, L]),
                                        Z: sample_Z([batch_size, condition_length, Z_dim_c]),
                                        Noise_std: np.sqrt(1 / (2 * R * EbNo_train))
                                        })
        print("Generated Channel Evaluation:", "Step " + str(step) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss) + ", Training Accuracy= " + \
              "{:.3f}".format(acc))

        loss, acc = sess.run([loss_receiver_R, accuracy_R],
                             feed_dict={X: test_data,
                                        h_i: generate_channel_parts(channel_PDP, [len(test_data), L]),
                                        h_r: generate_channel_parts(channel_PDP, [len(test_data), L]),
                                        Noise_std: np.sqrt(1 / (2 * R * EbNo_train))})
        print("Real Channel Evaluation:", "Step " + str(step) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss) + ", Test Accuracy= " + \
              "{:.3f}".format(acc))

        loss, acc = sess.run([loss_receiver_G, accuracy_G],
                             feed_dict={X: test_data,
                                        h_i: generate_channel_parts(channel_PDP, [len(test_data), L]),
                                        h_r: generate_channel_parts(channel_PDP, [len(test_data), L]),
                                        Z: sample_Z([len(test_data), condition_length, Z_dim_c]),
                                        Noise_std: np.sqrt(1 / (2 * R * EbNo_train))
                                        })
        print("Generated Channel Evaluation:", "Step " + str(step) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss) + ", Test Accuracy= " + \
              "{:.3f}".format(acc))

        EbNodB_range = np.arange(0, 30)
        ber = np.ones(len(EbNodB_range))
        wer = np.ones(len(EbNodB_range))
        for n in range(0, len(EbNodB_range)):
            EbNo = 10.0 ** (EbNodB_range[n] / 10.0)
            ber[n], wer[n] = sess.run([accuracy_R, WER_R],
                                      feed_dict={X: test_data, Noise_std: (np.sqrt(1 / (2 * R * EbNo * (1+CP/N)))),      # E2E use zero padding, (1+CP/N) is used for a fair comparison with OFMD with CP,
                                                 h_i: generate_channel_parts(channel_PDP, [len(test_data), L]),
                                                 h_r: generate_channel_parts(channel_PDP, [len(test_data), L]),
                                                 })
            print('SNR:', EbNodB_range[n], 'BER:', ber[n], 'WER:', wer[n])
        print(ber)
        print(wer)




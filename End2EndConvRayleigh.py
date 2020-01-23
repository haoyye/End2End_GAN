from __future__ import division
import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

''' This file aims to solve the end to end communication problem in Rayleigh fading channel '''
''' The condition of channel GAN is the encoding and information h '''
''' We should compare with baseline that equalizor of Rayleigh fading'''

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
        layer_4_normalized = tf.scalar_mul(tf.sqrt(tf.cast(block_length/2, tf.float32)),
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
        return Decoding_logit, Decoding_prob


def sample_Z(sample_size):
    ''' Sampling the generation noise Z from normal distribution '''
    return np.random.normal(size=sample_size)


def sample_uniformly(sample_size):
    return np.random.randint(size=sample_size, low=-15, high=15) / 10


def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


def Rayleigh_noise_layer(input_layer, h_r, h_i, std):
    h_complex = tf.complex(real=h_r, imag=h_i)
    input_layer_real = input_layer[:, :, 0]
    input_layer_imag = input_layer[:, :, 1]
    input_layer_complex = tf.complex(real=input_layer_real, imag=input_layer_imag)
    # input_layer_complex = tf.reshape(input_layer_complex, [-1, block_length, 1])
    noise = tf.cast(tf.random_normal(shape=tf.shape(input_layer_complex), mean=0.0, stddev=std, dtype=tf.float32),
                    tf.complex64)
    noise = tf.complex(
        real=tf.random_normal(shape=tf.shape(input_layer_complex), mean=0.0, stddev=std, dtype=tf.float32),
        imag=tf.random_normal(shape=tf.shape(input_layer_complex), mean=0.0, stddev=std, dtype=tf.float32))
    output_complex = tf.add(tf.multiply(h_complex, input_layer_complex), noise)
    output_complex_reshape = tf.reshape(output_complex, [-1, block_length, 1])
    print("Shape of the output complex", output_complex, output_complex_reshape)

    # print("shape of the complex matrix", input_layer_complex, output_complex, tf.concat([tf.real(output_complex), tf.imag(output_complex)], -1))
    return tf.concat([tf.real(output_complex_reshape), tf.imag(output_complex_reshape)], -1)


def sample_h(sample_size):
    return np.random.normal(size=sample_size) / np.sqrt(2.)


""" Start of the Main function """

''' Building the Graph'''
batch_size = 512
block_length = 128
Z_dim_c = 16
learning_rate = 1e-4

X = tf.placeholder(tf.float32, shape=[None, block_length, 1])
E = encoding(X)
Z = tf.placeholder(tf.float32, shape=[None, block_length, Z_dim_c])
Noise_std = tf.placeholder(tf.float32, shape=[])

h_r = tf.placeholder(tf.float32, shape=[None, 1])
h_i = tf.placeholder(tf.float32, shape=[None, 1])

h_r_noise = tf.add(h_r, tf.random_normal(shape=tf.shape(h_r), mean=0.0, stddev=Noise_std, dtype=tf.float32))
h_i_noise = tf.add(h_i, tf.random_normal(shape=tf.shape(h_i), mean=0.0, stddev=Noise_std, dtype=tf.float32))

#Channel_info = tf.tile(tf.concat([tf.reshape(h_r_noise, [-1, 1, 1]), tf.reshape(h_i_noise, [-1, 1, 1])], -1), [1, block_length, 1])
Channel_info = tf.tile(tf.concat([tf.reshape(h_r, [-1, 1, 1]), tf.reshape(h_i, [-1, 1, 1])], -1), [1, block_length, 1])
Conditions = tf.concat([E, Channel_info], axis=-1)

G_sample = generator_conditional(Z, Conditions)
R_sample = Rayleigh_noise_layer(E, h_r, h_i, Noise_std)

R_decodings_logit, R_decodings_prob = decoding(R_sample, Channel_info)
G_decodings_logit, G_decodings_prob = decoding(G_sample, Channel_info)

encodings_uniform_generated = tf.placeholder(tf.float32, shape=[None, block_length, 2])
Conditions_uniform = tf.concat([encodings_uniform_generated, Channel_info], axis=-1)

print("shapes G and R and channel info", G_sample, R_sample, encodings_uniform_generated)
G_sample_uniform = generator_conditional(Z, Conditions_uniform)
R_sample_uniform = Rayleigh_noise_layer(encodings_uniform_generated, h_r, h_i, Noise_std)
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
# Set up solvers

D_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(D_loss, var_list=Disc_vars)
G_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(G_loss, var_list=Gen_vars)

''' WGAN-GP'''
'''
lambdda = 1
alpha = tf.random_uniform(shape=tf.shape(X), minval=0.,maxval=1.)
differences = G_sample - X
interpolates = X + (alpha * differences)
_, D_inter = discriminator_condintional(interpolates, encodings_uniform_generated )
gradients = tf.gradients(D_inter, [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes-1.0)**2)
D_loss += lambdda * gradient_penalty

D_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(D_loss, var_list = Disc_vars)
G_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(G_loss, var_list = Gen_vars)

'''

loss_receiver_R = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=R_decodings_logit, labels=X))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
Rx_solver = optimizer.minimize(loss_receiver_R, var_list=Rx_vars)
loss_receiver_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=G_decodings_logit, labels=X))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
Tx_solver = optimizer.minimize(loss_receiver_G, var_list=Tx_vars)
accuracy_R = tf.reduce_mean(tf.cast((tf.abs(R_decodings_prob - X) > 0.5), tf.float32))
accuracy_G = tf.reduce_mean(tf.cast((tf.abs(G_decodings_prob - X) > 0.5), tf.float32))
WER_R = 1 - tf.reduce_mean(tf.cast(tf.reduce_all(tf.abs(R_decodings_prob-X)<0.5, 1),tf.float32))
init = tf.global_variables_initializer()
number_steps_receiver = 5000
number_steps_channel = 5000
number_steps_transmitter = 5000
display_step = 100
batch_size = 320
number_iterations = 1000  # in each iteration, the receiver, the transmitter and the channel will be updated

EbNo_train = 20.
EbNo_train = 10. ** (EbNo_train / 10.)

EbNo_train_GAN = 35.
EbNo_train_GAN = 10. ** (EbNo_train_GAN / 10.)

EbNo_test = 15.
EbNo_test = 10. ** (EbNo_test / 10.)

R = 0.5

def generate_batch_data(batch_size):
    global start_idx, data
    if start_idx + batch_size >= N_training:
        start_idx = 0
        data = np.random.binomial(1, 0.5, [N_training, block_length, 1])
    batch_x = data[start_idx:start_idx + batch_size]
    start_idx += batch_size
    #print("start_idx", start_idx)
    return batch_x




N_training = int(1e6)
data = np.random.binomial(1, 0.5, [N_training, block_length, 1])
N_val = int(1e4)
val_data = np.random.binomial(1, 0.5, [N_val, block_length, 1])
N_test = int(1e4)
test_data = np.random.binomial(1, 0.5, [N_test, block_length, 1])

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
saver = tf.train.Saver()
load_pretrain_model = False
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    start_idx = 0
    if load_pretrain_model and tf.train.latest_checkpoint('./Rayleigh_GAN_Models/'):
        saver.restore(sess, tf.train.latest_checkpoint('./Rayleigh_GAN_Models/'))
    for iteration in range(number_iterations):
        number_steps_transmitter += 5000
        number_steps_receiver += 5000
        number_steps_channel += 2000
        print("iteration is ", iteration)
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
                                                 h_i: sample_h([batch_size, 1]),
                                                 h_r: sample_h([batch_size, 1]),
                                                 Z: sample_Z([batch_size, block_length, Z_dim_c]),
                                                 Noise_std: (np.sqrt(1 / (2 * R * EbNo_train_GAN)))})
            _, G_loss_curr = sess.run([G_solver, G_loss],
                                      feed_dict={encodings_uniform_generated: input_data,
                                                 h_i: sample_h([batch_size, 1]),
                                                 h_r: sample_h([batch_size, 1]),
                                                 Z: sample_Z([batch_size, block_length, Z_dim_c]),
                                                 Noise_std: (np.sqrt(1 / (2 * R * EbNo_train_GAN)))})

        ''' =========== Training the Transmitter ==== '''
        for step in range(number_steps_transmitter):
            if step % 100 == 0:
                print("Training transmitter, step is ", step)
            batch_x = generate_batch_data(batch_size)
            sess.run(Tx_solver, feed_dict={X: batch_x, Z: sample_Z([batch_size, block_length, Z_dim_c]),
                                           h_i: sample_h([batch_size, 1]),
                                           h_r: sample_h([batch_size, 1]),
                                           Noise_std: (np.sqrt(1 / (2 * R * EbNo_train)))
                                           })

        ''' ========== Training the Receiver ============== '''

        for step in range(number_steps_receiver):
            if step % 100 == 0:
                print("Training receiver, step is ", step)
            batch_x = generate_batch_data(batch_size)
            sess.run(Rx_solver, feed_dict={X: batch_x,
                                           h_i: sample_h([batch_size, 1]),
                                           h_r: sample_h([batch_size, 1]),
                                           Noise_std: (np.sqrt(1 / (2 * R * EbNo_train)))})

        '''  ----- Testing ----  '''

        loss, acc = sess.run([loss_receiver_R, accuracy_R],
                             feed_dict={X: batch_x,
                                        h_i: sample_h([batch_size, 1]),
                                        h_r: sample_h([batch_size, 1]),
                                        Noise_std: np.sqrt(1 / (2 * R * EbNo_train))})
        print("Real Channel Evaluation:", "Step " + str(step) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss) + ", Training Accuracy= " + \
              "{:.3f}".format(acc))

        loss, acc = sess.run([loss_receiver_G, accuracy_G],
                             feed_dict={X: batch_x,
                                        h_i: sample_h([batch_size, 1]),
                                        h_r: sample_h([batch_size, 1]),
                                        Z: sample_Z([batch_size, block_length, Z_dim_c]),
                                        Noise_std: np.sqrt(1 / (2 * R * EbNo_train))
                                        })
        print("Generated Channel Evaluation:", "Step " + str(step) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss) + ", Training Accuracy= " + \
              "{:.3f}".format(acc))
        saver.save(sess, 'Rayleigh_GAN_Models_Real_H/'+str(iteration)+'.ckpt')
        EbNodB_range = np.arange(0, 30)
        ber = np.ones(len(EbNodB_range))
        wer = np.ones(len(EbNodB_range))
        for n in range(0, len(EbNodB_range)):
            EbNo = 10.0 ** (EbNodB_range[n] / 10.0)
            ber[n], wer[n] = sess.run([accuracy_R, WER_R],
                              feed_dict={X: test_data, Noise_std: (np.sqrt(1 / (2 * R * EbNo))),
                                         h_i: sample_h([len(test_data), 1]),
                                         h_r: sample_h([len(test_data), 1]),
                                         })
            print('SNR:', EbNodB_range[n], 'BER:', ber[n], 'WER:', wer[n])

        print(ber)
        print(wer)
        plt.plot(EbNodB_range, ber, 'bo', label='Learning_Based_Rayleigh fading')
        # plt.plot(list(EbNodB_range), ber_theory, 'ro-',label='BPSK BER')
        plt.yscale('log')
        plt.xlabel('SNR Range')
        plt.ylabel('Block Error Rate')
        plt.grid()
        plt.legend(loc='upper right', ncol=1)
        plt.show()

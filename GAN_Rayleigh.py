import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.neighbors import NearestNeighbors


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

""" This file is trying to simulate the Rayleigh channel without any channel information"""

tf.set_random_seed(100)
np.random.seed(100)

def KL_estimator(X, Y, k):
    n, m = len(X), len(Y)
    d = float(X.shape[1])
    D = np.log(m / (n - 1))
    X_neighbourhood = NearestNeighbors(k+1, 10).fit(X)
    Y_neighbourhood = NearestNeighbors(k, 10).fit(Y)
    for p1 in X:
        s1_distances, indices = X_neighbourhood.kneighbors([p1], k+1)
        s2_distances, indices = Y_neighbourhood.kneighbors([p1], k)
        rho = s1_distances[0][-1]
        nu = s2_distances[0][-1]
        D += (d/n)*np.log(nu/rho)
    return D
def generator_conditional(z, conditioning):  # need to change the structure
    z_combine = tf.concat([z, conditioning], 1)
    G_h1 = tf.nn.relu(tf.matmul(z_combine, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_h3 = tf.nn.relu(tf.matmul(G_h2, G_W3) + G_b3)
    G_logit = tf.matmul(G_h3, G_W4) + G_b4
    return G_logit


def discriminator_conditional(X,  conditioning):  # need to change the structure
    z_combine = tf.concat([X, conditioning], 1)
    D_h1_real = tf.nn.relu(tf.matmul(z_combine / 4, D_W1) + D_b1)
    #D_h2_real = tf.reduce_mean(tf.nn.relu(tf.matmul(D_h1_real, D_W2) + D_b2), axis=0, keep_dims=True)
    D_h2_real = tf.nn.relu(tf.matmul(D_h1_real, D_W2) + D_b2)
    D_h3_real = tf.nn.relu(tf.matmul(D_h2_real, D_W3) + D_b3)
    D_logit = tf.matmul(D_h3_real, D_W4) + D_b4
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob, D_logit


def sample_Z(sample_size):
    ''' Sampling the generation noise Z from normal distribution '''
    return np.random.normal(size=sample_size)

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


number = 200
h_r = np.random.normal(scale=np.sqrt(2) / 2, size=number)
h_i = np.random.normal(scale=np.sqrt(2) / 2, size=number)
h_complex = h_r + 1j * h_i


def generate_real_samples_with_labels_Rayleigh(number=100):
    h_r = np.random.normal(scale=np.sqrt(2) / 2, size=number)
    h_i = np.random.normal(scale=np.sqrt(2) / 2, size=number)
    h_complex = h_r + 1j * h_i
    labels_index = np.random.choice(len(mean_set_QAM), number)
    data = mean_set_QAM[labels_index]
    received_data = h_complex * data
    received_data = np.hstack(
        (np.real(received_data).reshape(len(data), 1), np.imag(received_data).reshape(len(data), 1)))
    gaussion_random = np.random.multivariate_normal([0, 0], [[0.01, 0], [0, 0.01]], number).astype(np.float32)
    received_data = received_data + gaussion_random
    conditioning = np.hstack((np.real(data).reshape(len(data), 1), np.imag(data).reshape(len(data), 1),
                              h_r.reshape(len(data), 1), h_i.reshape(len(data), 1))) / 3
    return received_data, conditioning



""" ==== Here is the main function ==== """
mean_set_QAM = np.asarray([-3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j, -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j,
                           1 - 3j, 1 - 1j, 1 + 1j, 1 + 3j, 3 - 3j, 3 - 1j, 3 + 1j, 3 + 3j
                           ], dtype=np.complex64)
batch_size = 512
condition_depth = 2
condition_dim = 4
Z_dim = 16
model = 'ChannelGAN_Rayleigh_'
data_size = 10000
data, one_hot_labels = generate_real_samples_with_labels_Rayleigh(data_size)
D_W1 = tf.Variable(xavier_init([2 + condition_dim, 32]))
D_b1 = tf.Variable(tf.zeros(shape=[32]))
D_W2 = tf.Variable(xavier_init([32, 32]))
D_b2 = tf.Variable(tf.zeros(shape=[32]))
D_W3 = tf.Variable(xavier_init([32, 32]))
D_b3 = tf.Variable(tf.zeros(shape=[32]))
D_W4 = tf.Variable(xavier_init([32, 1]))
D_b4 = tf.Variable(tf.zeros(shape=[1]))
theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3, D_W4, D_b4]
G_W1 = tf.Variable(xavier_init([Z_dim + condition_dim, 128]))
G_b1 = tf.Variable(tf.zeros(shape=[128]))
G_W2 = tf.Variable(xavier_init([128, 128]))
G_b2 = tf.Variable(tf.zeros(shape=[128]))
G_W3 = tf.Variable(xavier_init([128, 128]))
G_b3 = tf.Variable(tf.zeros(shape=[128]))
G_W4 = tf.Variable(xavier_init([128, 2]))
G_b4 = tf.Variable(tf.zeros(shape=[2]))
theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3, G_W4, G_b4]
save_fig_path = model
if not os.path.exists(save_fig_path):
    os.makedirs(save_fig_path)
R_sample = tf.placeholder(tf.float32, shape=[None, 2])
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])
Condition = tf.placeholder(tf.float32, shape=[None, condition_dim])
G_sample = generator_conditional(Z, Condition)
D_prob_real, D_logit_real = discriminator_conditional(R_sample, Condition)
D_prob_fake, D_logit_fake = discriminator_conditional(G_sample, Condition)

#Disc_vars = [v for v in tf.trainable_variables() if v.name.startswith('discriminator')]
#Gen_vars = [v for v in tf.trainable_variables() if v.name.startswith('generator')]

#D_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(D_loss, var_list=theta_D)
#G_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(G_loss, var_list=theta_G)


''' WGAN-GP'''
D_loss = tf.reduce_mean(D_logit_fake) - tf.reduce_mean(D_logit_real)
G_loss = -1 * tf.reduce_mean(D_logit_fake)
lambdda = 5
alpha = tf.random_uniform(shape=tf.shape(R_sample), minval=0., maxval=1.)
differences = G_sample - R_sample
interpolates = R_sample + (alpha * differences)
_, D_inter = discriminator_conditional(interpolates, Condition)
gradients = tf.gradients(D_inter, [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes - 1.0) ** 2)
D_loss += lambdda * gradient_penalty
D_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(G_loss, var_list=theta_G)

'''WGAN with weight clipping'''

'''

D_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(D_loss, var_list=Disc_vars)
G_solver = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(G_loss, var_list=Gen_vars)

clip_ops = []
for var in Disc_vars:
    clip_bounds = [-.01, .01]
    clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
    clip_disc_weights = tf.group(*clip_ops)

'''

sess = tf.Session()
sess.run(tf.global_variables_initializer())

save_fig_path = model+"images"
if not os.path.exists(save_fig_path):
    os.makedirs(save_fig_path)
i = 0
plt.figure(figsize=(5, 5))
plt.plot(data[:1000, 0], data[:1000, 1], 'b.')
axes = plt.gca()
axes.set_xlim([-4, 4])
axes.set_ylim([-4, 4])
plt.title('True data distribution')
plt.savefig(save_fig_path + '/real.png', bbox_inches='tight')
np_samples = []
plot_every = 1000
plt.figure(figsize=(5, 5))
xmax = 4
saver = tf.train.Saver()
KL_test_step = 1

KL_distance = []
iter_check_index = []
for it in range(750000):
    start_idx = it * batch_size % data_size
    if start_idx + batch_size >= len(data):
        continue
    X_mb = data[start_idx:start_idx + batch_size, :]
    one_hot_labels_mb = one_hot_labels[start_idx:start_idx + batch_size, :]
    for d_idx in range(10):
        _, D_loss_curr = sess.run([D_solver, D_loss],
                                  feed_dict={R_sample: X_mb, Z: sample_Z((batch_size, Z_dim)),
                                             Condition: one_hot_labels_mb})
        # print("Inner loop Losses: D are", D_loss_curr)
    # print("start_idx is", start_idx, "shape of one hot label", one_hot_labels_mb.shape, X_mb.shape )
    #_, D_loss_curr = sess.run([D_solver, D_loss],
    #                          feed_dict={R_sample: X_mb, Z: sample_Z((batch_size, Z_dim)), Condition: one_hot_labels_mb})
    _, G_loss_curr = sess.run([G_solver, G_loss],
                              feed_dict={R_sample: X_mb, Z: sample_Z((batch_size, Z_dim)), Condition: one_hot_labels_mb})

    #save_path = saver.save(sess, './Models/ChannelGAN_model_step_' + str(it) + '.ckpt')

    if (it + 1) % plot_every == 0:
        save_path = saver.save(sess, './Models/ChannelGAN_model_step_' + str(it) + '.ckpt')

        # Calculate the KL distance
        def Compute_KL():
            number = 5000
            iter_channel = 10
            distance_list = np.zeros(iter_channel)
            for channel_idx in range(iter_channel):  # average of 100 channel realization
                h_r = np.random.normal(scale=np.sqrt(2) / 2)
                h_i = np.random.normal(scale=np.sqrt(2) / 2)
                h_r = np.tile(h_r, number)
                h_i = np.tile(h_i, number)
                idx = np.random.randint(len(mean_set_QAM))
                labels_index = np.tile(idx, number)
                h_complex = h_r + 1j * h_i
                # labels_index = np.random.choice(len(mean_set_QAM), number)
                data_t = mean_set_QAM[labels_index]
                transmit_data = h_complex * data_t
                # print("shapes", transmit_data.shape, h_complex.shape, data_t.shape)
                transmit_data = np.hstack((np.real(transmit_data).reshape(len(transmit_data), 1),
                                           np.imag(transmit_data).reshape(len(transmit_data), 1)))
                gaussion_random = np.random.multivariate_normal([0, 0], [[0.01, 0], [0, 0.01]], number).astype(
                    np.float32)
                received_data = transmit_data + gaussion_random
                conditioning = np.hstack(
                    (np.real(data_t).reshape(len(data_t), 1), np.imag(data_t).reshape(len(data_t), 1),
                     h_r.reshape(len(data_t), 1), h_i.reshape(len(data_t), 1)))/3

                samples_component = sess.run(G_sample,
                                             feed_dict={Z: sample_Z((number, Z_dim)), Condition: conditioning})

                distance_list[channel_idx] = KL_estimator(samples_component, received_data, 10)
                if channel_idx%20 ==0:
                    plt.clf()
                    plt.plot(samples_component[0:200:10, 0], samples_component[0:200:10, 1], 'k.')
                    plt.plot(received_data[0:200:10, 0], received_data[0:200:10, 1], 'b*')
                    plt.savefig(save_fig_path + '/plot_samples_'+str(channel_idx)+'{}.png'.format(str(it).zfill(3)), bbox_inches='tight')



            return np.mean(distance_list)
        print("start to calculate KL distance")


        KL_distance.append(Compute_KL())
        iter_check_index.append(it)
        print("iteration idx", iter_check_index)
        print("KL_distance", KL_distance)
        plt.clf()
        plt.plot(iter_check_index,KL_distance, 'bo-')
        plt.savefig(save_fig_path + '/plot_{}.png'.format(str(it).zfill(3)), bbox_inches='tight')
        '''

        print("Start Plotting")
        colors = ['b.', 'r+', 'm.', 'c.', 'k.', 'g.', 'y.', 'm.', \
                  'bo', 'ro', 'mo', 'co', 'ko', 'go', 'yo', 'bo']
        colors = ['b.', 'b+', 'bx', 'b^', 'b^', 'bx', 'b+', 'b.', \
                  'b.', 'b+', 'bx', 'b^', 'b^', 'bx', 'b+', 'b.']
        plt.clf()
        samples = np.array([])
        for channel_idx in range(10):
            plt.clf()
            number = 20  #
            h_r = np.random.normal(scale=np.sqrt(2) / 2)
            h_i = np.random.normal(scale=np.sqrt(2) / 2)
            h_r = np.tile(h_r, number)
            h_i = np.tile(h_i, number)
            for idx in range(len(mean_set_QAM)):
                labels_index = np.tile(idx, number)
                h_complex = h_r + 1j * h_i
                # labels_index = np.random.choice(len(mean_set_QAM), number)
                data_t = mean_set_QAM[labels_index]
                transmit_data = h_complex * data_t
                # print("shapes", transmit_data.shape, h_complex.shape, data_t.shape)
                transmit_data = np.hstack((np.real(transmit_data).reshape(len(transmit_data), 1),
                                           np.imag(transmit_data).reshape(len(transmit_data), 1)))
                gaussion_random = np.random.multivariate_normal([0, 0], [[0.03, 0], [0, 0.03]], number).astype(
                    np.float32)
                received_data = transmit_data + gaussion_random
                conditioning = np.hstack(
                    (np.real(data_t).reshape(len(data_t), 1), np.imag(data_t).reshape(len(data_t), 1),
                     h_r.reshape(len(data_t), 1), h_i.reshape(len(data_t), 1))) /3
                samples_component = sess.run(G_sample, feed_dict={Z: sample_Z((number, Z_dim)), Condition: conditioning})
                plt.plot(samples_component[:, 0], samples_component[:, 1], colors[idx])
                plt.plot(transmit_data[:, 0], transmit_data[:, 1], colors[idx])
                #plt.plot(samples_component[:, 0], samples_component[:, 1], 'k.')
                #plt.plot(transmit_data[:, 0], transmit_data[:, 1], 'b*')
            axes = plt.gca()
            axes.set_xlim([-4, 4])
            axes.set_ylim([-4, 4])
            xlabel = r'$Re\{y_n\}$'
            ylabel = r'$Imag\{y_n\}$'
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.show()
            plt.savefig( save_fig_path + '/' + str(channel_idx) + '_{}_noise_1.eps'.format(str(i).zfill(3)),
                        bbox_inches='tight')
            plt.savefig(save_fig_path + '/' + str(channel_idx) + '_{}_noise_1.png'.format(str(i).zfill(3)),
                        bbox_inches='tight')

        axes.set_xlim([-4, 4])
        axes.set_ylim([-4, 4])
        plt.title('Iter: {}, loss(D): {:2.2f}, loss(G):{:2.2f}'.format(it + 1, D_loss_curr, G_loss_curr))
        plt.savefig(save_fig_path + '/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')

        i += 1
        '''





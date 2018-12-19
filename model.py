import os, imageio
import time
import math
from glob import glob
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils.wavelet_img import *
from ops import *

class CWGAN(object):
    def __init__(self, sess, y_dim, output_dir, checkpoint_dir,
                 img_height=64, img_width=64,  batch_size=64,
                 z_dim=100, c_dim=1):

        self.sess = sess
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width

        self.y_dim = y_dim
        self.z_dim = z_dim
        self.c_dim = c_dim

        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.wavelet_imgs = Wav_img()

        self.build_model()

    def build_model(self):
        self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
        image_dims = [self.img_height, self.img_width, self.c_dim]

        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')

        inputs = self.inputs

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        self.G = self.generator(self.z, self.y, reuse=False)
        self.D, self.D_logits = self.discriminator(inputs, self.y, reuse=False)

        self.D_, self.D_logits_ = self.discriminator(self.G, self.y, reuse=True)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            x = conv_cond_concat(image, yb)

            h0 = lrelu(conv2d(x, 5, name='d_h0_conv'))
            h0 = conv_cond_concat(h0, yb)

            h1 = lrelu(self.d_bn1(conv2d(h0, 68, name='d_h1_conv')))
            h1 = tf.reshape(h1, [self.batch_size, -1])
            h1 = concat([h1, y], 1)

            h2 = lrelu(self.d_bn2(linear(h1, 1024, 'd_h2_lin')))
            h2 = concat([h2, y], 1)

            h3 = linear(h2, 1, 'd_h3_lin')

            return tf.nn.sigmoid(h3), h3

    def generator(self, z, y=None, reuse=False, train=True):
        with tf.variable_scope("generator") as scope:
            if reuse:
                scope.reuse_variables()
            s_h, s_w = self.img_height, self.img_width
            s_h2, s_h4 = int(s_h / 2), int(s_h / 4)
            s_w2, s_w4 = int(s_w / 2), int(s_w / 4)

            # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = concat([z, y], 1)

            h0 = tf.nn.relu(
                self.g_bn0(linear(z, 1024, 'g_h0_lin'), train=train))
            h0 = concat([h0, y], 1)

            h1 = tf.nn.relu(self.g_bn1(
                linear(h0, 128 * s_h4 * s_w4, 'g_h1_lin'), train=train))
            h1 = tf.reshape(h1, [self.batch_size, s_h4, s_w4, 128])

            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1,
                                                [self.batch_size, s_h2, s_w2, 128], name='g_h2'), train=train))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.tanh(
                deconv2d(h2, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3'))

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    @property
    def model_dir(self):
        return "{}_{}_{}".format(
            self.batch_size,
            self.img_height, self.img_width)

    def save(self, checkpoint_dir, step):
        model_name = "CGAN.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def train(self, config):
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        tf.global_variables_initializer().run()

        self.g_sum = merge_summary([self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = merge_summary([self.d_loss_real_sum, self.d_loss_sum])
        self.writer = SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(64, self.z_dim))
        _, sample_labels = self.wavelet_imgs.next_batch(64)

        counter = 1
        start_time = time.time()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        while self.wavelet_imgs._epochs_completed < config.epoch:

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                    .astype(np.float32)

                batch_images, batch_labels = self.wavelet_imgs.next_batch(self.batch_size)

                # Update D network
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                                               feed_dict={
                                                   self.inputs: batch_images,
                                                   self.z: batch_z,
                                                   self.y: batch_labels,
                                               })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={
                                                   self.z: batch_z,
                                                   self.y: batch_labels,
                                               })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                                               feed_dict={self.z: batch_z, self.y: batch_labels})
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({
                    self.z: batch_z,
                    self.y: batch_labels
                })
                errD_real = self.d_loss_real.eval({
                    self.inputs: batch_images,
                    self.y: batch_labels
                })
                errG = self.g_loss.eval({
                    self.z: batch_z,
                    self.y: batch_labels
                })

                counter += 1
                print("Epoch: [%2d/%2d] [%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (self.wavelet_imgs._epochs_completed, config.epoch, counter,
                         time.time() - start_time, errD_fake + errD_real, errG))

                if counter % 100 == 0:
                        samples = self.sess.run(self.generator(self.z, self.y, reuse=True, train=False),
                                                feed_dict={self.z: sample_z,
                                                           self.y: sample_labels})

                        for i, gen_img in enumerate(samples):
                            imageio.imsave(os.path.join(self.output_dir, './train_{:02d}_%s.png'.format(counter) % i), gen_img)

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)


    def sample(self, sample_label, plot=False, save_np=True, sample_times=1):
        assert isinstance(sample_label, str)
        assert sample_label in one_hot
        sample_labels = np.tile(np.array(one_hot[sample_label]), [self.batch_size, 1]).astype(np.float32)

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        ranges = np.tile(self.wavelet_imgs.ranges, (self.batch_size, 1, 1, 1))
        means = np.tile(self.wavelet_imgs.mean_arr, (self.batch_size, 1, 1, 1))

        if save_np:
            data = []
            norm_data = []
            for t in range(sample_times):
                sample_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                norm_samples = self.sess.run(self.generator(self.z, self.y, reuse=True, train=False),
                                        feed_dict={self.z: sample_z,
                                                   self.y: sample_labels})
                # TODO 还原norm
                samples = norm_samples * ranges + means

                data.append(samples)
                norm_data.append(norm_samples)

            data = np.concatenate(data, axis=0)
            norm_data = np.concatenate(norm_data, axis=0)

            if not os.path.exists(os.path.join(self.output_dir, 'gan_generate_data')):
                os.makedirs(os.path.join(self.output_dir, 'gan_generate_data'))
            np.save(os.path.join(self.output_dir, 'gan_generate_data', '%s_%s.npy' % (sample_label, sample_times)), data)
            np.save(os.path.join(self.output_dir, 'gan_generate_data', 'norm_%s_%s.npy' % (sample_label, sample_times)), norm_data)



        if plot:
            sample_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
            norm_samples = self.sess.run(self.generator(self.z, self.y, reuse=True, train=False),
                                    feed_dict={self.z: sample_z,
                                               self.y: sample_labels})

            # TODO 还原norm
            samples = norm_samples * ranges + means

            plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.1, hspace=0.1)
            for i, gen_img in enumerate(samples[0: 8]):
                # imageio.imsave(os.path.join(self.output_dir, './train_{:02d}_%s.png'.format(counter) % i), gen_img)
                plt.subplot(1, 8, i+1)
                plt.axis('off')
                print(gen_img.reshape([self.img_width, self.img_height]))
                print(gen_img.reshape([self.img_width, self.img_height]).shape)
                plt.imshow(gen_img.reshape([self.img_width, self.img_height]), cmap='gray')

            plt.savefig(os.path.join(self.output_dir, '%s.png' % sample_label), format='png')

            for i, gen_img in enumerate(norm_samples[0: 8]):
                # imageio.imsave(os.path.join(self.output_dir, './train_{:02d}_%s.png'.format(counter) % i), gen_img)
                plt.subplot(1, 8, i + 1)
                plt.axis('off')
                print(gen_img.reshape([self.img_width, self.img_height]))
                print(gen_img.reshape([self.img_width, self.img_height]).shape)
                plt.imshow(gen_img.reshape([self.img_width, self.img_height]), cmap='gray')

            plt.savefig(os.path.join(self.output_dir, 'norm_%s.png' % sample_label), format='png')
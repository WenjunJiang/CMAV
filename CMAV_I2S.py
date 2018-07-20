s#-*- coding: utf-8 -*-
from __future__ import division
import os
import time
import tensorflow as tf
import numpy as np
from glob import glob

from ops import *
from utils import *

class I2S(object):
    def __init__(self, sess, epoch, batch_size, z_dim, dataset_name, checkpoint_dir, result_dir, log_dir, test_dir):
        self.sess = sess
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.result_dir = result_dir
        self.log_dir = log_dir
        self.test_dir = test_dir
        self.epoch = epoch
        self.batch_size = batch_size
        self.model_name = "I2S"     # name for checkpoint

        if dataset_name == 'Sub-URMP':
            # parameters
            self.input_height = 108
            self.input_width = 130
            self.output_height = 64
            self.output_width = 64

            self.z_dim = z_dim         # dimension of noise-vector
            self.y_dim = 13         # dimension of code-vector (label)
            self.c_dim = 3

            # train
            self.learning_rate = 0.0001
            self.beta1 = 0.5

            # test
            self.sample_num = 64  # number of generated images to be saved

            #train iter
            self.train_iter = 0

            # load mnist
            self.data_X, self.data_S, self.data_MIS, self.data_y = load_Sub(self.dataset_name, self.y_dim)

            # get number of batches for a single epoch
            self.num_batches = len(self.data_X) // self.batch_size
        else:
            raise NotImplementedError

    def encoder(self, x, is_training=True, reuse=False):
        with tf.variable_scope("encoder", reuse=reuse):
            net = lrelu(conv2d(x, 64, 4, 4, 2, 2, name='en_conv1'))
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='en_conv2'), is_training=is_training, scope='en_bn2'))
            net = lrelu(bn(conv2d(net, 256, 4, 4, 2, 2, name='en_conv3'), is_training=is_training, scope='en_bn3'))
            net = lrelu(bn(conv2d(net, 256, 4, 4, 1, 1, name='en_conv4'), is_training=is_training, scope='en_bn4'))
            net = lrelu(bn(conv2d(net, 512, 4, 4, 2, 2, name='en_conv5'), is_training=is_training, scope='en_bn5'))
            net = tf.reshape(net, [self.batch_size, -1])
            net = lrelu(bn(linear(net, 1024, scope='en_fc6'), is_training=is_training, scope='en_bn6'))
            out_classifier = lrelu(bn(linear(net, 128, scope='en_fc7'), is_training=is_training, scope='en_bn7'))
            out = lrelu(bn(linear(out_classifier, 64, scope='en_fc8'), is_training=is_training, scope='en_bn8'))
        return out, out_classifier

    def classifier(self, x, reuse=False):
        with tf.variable_scope("classifier", reuse=reuse):
            net = linear(x, 13, scope='c_fc1')
            prediction = tf.nn.softmax(net)
        return prediction

    def generator(self, z, x, is_training=True, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            z = concat([x, z], 1)
            net = tf.nn.relu(bn(linear(z, 512 * 4 * 4, scope='g_fc1'), is_training=is_training, scope='g_bn1'))
            net = tf.reshape(net, [self.batch_size, 4, 4, 512])
            net = tf.nn.relu(bn(deconv2d(net, [self.batch_size, 8, 8, 256], 4, 4, 2, 2, name='g_dc2'),
                                is_training=is_training, scope='g_bn2'))
            net = tf.nn.relu(bn(deconv2d(net, [self.batch_size, 8, 8, 256], 4, 4, 1, 1, name='g_dc3'),
                                is_training=is_training, scope='g_bn3'))
            net = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 16, 16, 128], 4, 4, 2, 2, name='g_dc4'))
            net = tf.nn.sigmoid(deconv2d(net, [self.batch_size, 32, 32, 64], 4, 4, 2, 2, name='g_dc5'))
            out = tf.nn.tanh(deconv2d(net, [self.batch_size, 64, 64, 3], 4, 4, 2, 2, name='g_dc6'))
        return out

    def discriminator(self, x_img, x_sound, is_training=True, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse):
            x_img = concat([x_img, x_img], 1)
            x_img = concat([x_img, x_img], 1)
            x_img = concat([x_img, x_img], 1)
            x_img = concat([x_img, x_img], 1)
            x_img = tf.reshape(x_img, [self.batch_size, 4, 4, 64])
            net = lrelu(conv2d(x_sound, 64, 4, 4, 2, 2, name='d_conv1'))
            net = lrelu(bn(conv2d(net, 128, 4, 4, 2, 2, name='d_conv2'), is_training=is_training, scope='d_bn2'))
            net = lrelu(bn(conv2d(net, 256, 4, 4, 2, 2, name='d_conv3'), is_training=is_training, scope='d_bn3'))
            net = lrelu(bn(conv2d(net, 256, 4, 4, 1, 1, name='d_conv4'), is_training=is_training, scope='d_bn4'))
            net = lrelu(bn(conv2d(net, 512, 4, 4, 2, 2, name='d_conv5'), is_training=is_training, scope='d_bn5'))
            net = conv_cond_concat(net, x_img)
            net = tf.reshape(net, [self.batch_size, -1])
            net = MinibatchLayer(32, 32, net, 'dB_fc6')
            net = lrelu(bn(linear(net, 1024, scope='d_fc7'), is_training=is_training, scope='d_bn7'))
            out_logit = linear(net, 1, scope='d_fc8')
            out = tf.nn.sigmoid(out_logit)
        return out, out_logit

    def mse_loss(self, pred, data):
        # tf.nn.l2_loss(pred - data) = sum((pred - data) ** 2) / 2
        loss_val = tf.sqrt(2 * tf.nn.l2_loss(pred - data)) / self.batch_size
        return loss_val

    def build_model(self):
        # some parameters
        image_dims = [self.output_height, self.output_width, self.c_dim]
        bs = self.batch_size

        """ Graph Input """
        # images
        self.inputs_img = tf.placeholder(tf.float32, [bs] + image_dims, name='real_images')

        self.inputs_sound = tf.placeholder(tf.float32, [bs] + image_dims, name='real_sounds')

        self.inputs_sound_mis = tf.placeholder(tf.float32, [bs] + image_dims, name='mis_sounds')

        # labels
        self.y = tf.placeholder(tf.float32, [bs, self.y_dim], name='y')

        # noises
        self.z = tf.placeholder(tf.float32, [bs, self.z_dim], name='z')

        """ Loss Function """

        En_Img, Cla_Img = self.encoder(self.inputs_img, is_training=True, reuse=False)

        G_sound = self.generator(self.z, En_Img, is_training=True, reuse=False)

        D_real, D_real_logits = self.discriminator(En_Img, self.inputs_sound, is_training=True, reuse=False)

        D_fake, D_fake_logits = self.discriminator(En_Img, G_sound, is_training=True, reuse=True)

        D_mis, D_mis_logits = self.discriminator(En_Img, self.inputs_sound_mis, is_training=True, reuse=True)

        prediction = self.classifier(Cla_Img, reuse=False)

        # Loss Discriminator
        d_loss_mis = tf.reduce_mean(self.mse_loss(D_mis_logits, tf.zeros_like(D_mis_logits)))
        d_loss_fake = tf.reduce_mean(self.mse_loss(D_fake_logits, tf.zeros_like(D_fake_logits)))
        d_loss_real = tf.reduce_mean(self.mse_loss(D_real_logits, tf.ones_like(D_real_logits)))

        # d_loss_real = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real)))
        #
        # d_loss_fake = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))
        #
        # d_loss_mis = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_mis_logits, labels=tf.zeros_like(D_mis)))

        self.d_loss = d_loss_real + (d_loss_fake + d_loss_mis)/2

        # Loss Generator
        self.g_loss = tf.reduce_mean(self.mse_loss(D_fake_logits, tf.ones_like(D_fake_logits)))

        # self.g_loss = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))

        # Loss Classifier
        self.classifier_loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(prediction), reduction_indices=[1]))

        """ Training """
        # divide trainable variables into a group for D and a group for G
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        #g_vars = [var for var in t_vars if ('g_' in var.name) or ('en_' in var.name)]
        g_vars = [var for var in t_vars if 'g_' in var.name]
        c_vars = [var for var in t_vars if ('c_' in var.name) or ('en_' in var.name)]

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.d_loss, var_list=d_vars)
            self.g_optim = tf.train.AdamOptimizer(self.learning_rate * 8, beta1=self.beta1) \
                .minimize(self.g_loss, var_list=g_vars)
            self.c_optim = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1) \
                .minimize(self.classifier_loss, var_list=c_vars)

        self.clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in d_vars]

        """" Testing """
        self.img, self.cla = self.encoder(self.inputs_img, is_training=False, reuse=True)
        self.prediction = self.classifier(self.cla, reuse=True)
        self.fake_images = self.generator(self.z, self.img, is_training=False, reuse=True)

        """ Summary """
        d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
        d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
        d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
        g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        c_loss_sum = tf.summary.scalar("c_loss", self.classifier_loss)

        # final summary operations
        self.g_sum = tf.summary.merge([d_loss_fake_sum, g_loss_sum])
        self.d_sum = tf.summary.merge([d_loss_real_sum, d_loss_sum])
        self.c_sum = tf.summary.merge([c_loss_sum])

    def train(self):

        # initialize all variables
        tf.global_variables_initializer().run()

        # graph inputs for visualize training results
        self.sample_z = np.random.normal(0, 1, size=(self.batch_size, self.z_dim))

        # saver to save model
        self.saver = tf.train.Saver()

        # restore check-point if it exits
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)
        if could_load:
            start_epoch = (int)(checkpoint_counter / self.num_batches) - 1
            start_batch_id = checkpoint_counter - (start_epoch+1) * self.num_batches
            counter = checkpoint_counter - 1112
            print(" [*] Load SUCCESS")
            if start_epoch == self.epoch:
                print('testing............')
                self.visualize_results_test(start_epoch)

        else:
            start_epoch = 0
            start_batch_id = 0
            counter = 1
            print(" [!] Load failed...")

        if start_epoch != self.epoch:
            # summary writer
            self.writer = tf.summary.FileWriter(self.log_dir + '/' + self.model_name, self.sess.graph)

        # loop for epoch
        start_time = time.time()

        if not could_load:
            print("pre_training Classifier")
            for idc in range(start_batch_id, self.num_batches):
                batch_files = self.data_X[idc * self.batch_size:(idc + 1) * self.batch_size]
                labels = self.data_y[idc * self.batch_size:(idc + 1) * self.batch_size]
                batch = [
                    get_image(batch_file,
                              input_height=self.input_height,
                              input_width=self.input_width,
                              resize_height=self.output_height,
                              resize_width=self.output_width,
                              ) for batch_file in batch_files]

                batch_images = np.array(batch).astype(np.float32)

                # update Classifer network
                _, summary_str, c_loss = self.sess.run([self.c_optim, self.c_sum, self.classifier_loss],
                                                       feed_dict={self.y: labels, self.inputs_img: batch_images})
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1

                if np.mod(counter, 30) == 0:
                    print("Epoch: [pre_training] [%4d/%4d] time: %4.4f, c_loss: %.8f"\
                          % (idc, self.num_batches, time.time() - start_time, c_loss))

            start_batch_id = 0
            # save model for final step
            self.save(self.checkpoint_dir, counter)

        for epoch in range(start_epoch, self.epoch):
            # get batch data
            for idx in range(start_batch_id, self.num_batches):
                batch_z = np.random.normal(0, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                batch_files = self.data_X[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_sounds_files = self.data_S[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_mis_files = self.data_MIS[idx * self.batch_size:(idx + 1) * self.batch_size]

                batch = [
                    get_image(batch_file,
                              input_height=self.input_height,
                              input_width=self.input_width,
                              resize_height=self.output_height,
                              resize_width=self.output_width,
                              ) for batch_file in batch_files]

                batch_images = np.array(batch).astype(np.float32)

                batch_S = [
                    get_image(batch_file_s,
                              input_height=0,
                              input_width=0,
                              resize_height=self.output_height,
                              resize_width=self.output_width,
                              crop=False
                              ) for batch_file_s in batch_sounds_files]

                batch_sounds = np.array(batch_S).astype(np.float32)
                batch_sounds = batch_sounds[:,:,:,:3]

                batch_MIS = [
                    get_image(batch_file_mis,
                              input_height=0,
                              input_width=0,
                              resize_height=self.output_height,
                              resize_width=self.output_width,
                              crop=False
                              ) for batch_file_mis in batch_mis_files]

                batch_mis_sounds = np.array(batch_MIS).astype(np.float32)
                batch_mis_sounds = batch_mis_sounds[:,:,:,:3]

                # update D network
                _, _, summary_str, d_loss = self.sess.run([self.d_optim, self.clip_D, self.d_sum, self.d_loss],
                                                        feed_dict={self.inputs_img: batch_images, self.inputs_sound: batch_sounds,
                                                                  self.inputs_sound_mis: batch_mis_sounds, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                # update G network
                _, summary_str, g_loss = self.sess.run([self.g_optim, self.g_sum, self.g_loss],
                                                       feed_dict={self.z: batch_z, self.inputs_img: batch_images})
                self.writer.add_summary(summary_str, counter)

                # display training status
                counter += 1

                if np.mod(counter, 5) == 0:
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"\
                          % (epoch, idx, self.num_batches, time.time() - start_time, d_loss, g_loss))

                # save training results for every 300 steps
                if np.mod(counter, 100) == 0:
                    samples = self.sess.run(self.fake_images,
                                            feed_dict={self.z: self.sample_z, self.inputs_img: batch_images})
                    tot_num_samples = min(self.sample_num, self.batch_size)
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    save_images(samples[:manifold_h * manifold_w, :, :, :], [manifold_h, manifold_w],
                                './' + check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name +
                                '_train_{:02d}_{:04d}.png'.format(epoch, idx))
            # After an epoch, start_batch_id is set to zero
            # non-zero value is only for the first epoch after loading pre-trained model
            start_batch_id = 0

            # save model
            self.save(self.checkpoint_dir, counter + 1112)

            # show temporal results
            self.visualize_results(epoch)
        # save model for final step
        self.save(self.checkpoint_dir, counter + 1112)

    def visualize_results(self, epoch):
        z_sample = np.random.normal(0, 1, [self.batch_size, self.z_dim]).astype(np.float32)

        batch_labels = self.data_y[5 * self.batch_size:(5 + 1) * self.batch_size]
        batch_files = self.data_X[5 * self.batch_size:(5 + 1) * self.batch_size]
        batch = [
            get_image(batch_file,
                      input_height=self.input_height,
                      input_width=self.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      ) for batch_file in batch_files]

        batch_images = np.array(batch).astype(np.float32)

        batch_sounds_files = self.data_S[5 * self.batch_size:(5 + 1) * self.batch_size]

        batch_S = [
            get_image(batch_file_s,
                      input_height=0,
                      input_width=0,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      crop=False
                      ) for batch_file_s in batch_sounds_files]

        batch_sounds = np.array(batch_S).astype(np.float32)
        batch_sounds = batch_sounds[:, :, :, :3]

        len_discrete_code_ply = 10

        prediction = self.sess.run(self.prediction, feed_dict={self.inputs_img:batch_images})

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample, self.inputs_img:batch_images})

        correct_prediction = np.equal(np.argmax(prediction, 1), np.argmax(batch_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = self.sess.run(accuracy, feed_dict={self.z: z_sample, self.inputs_img:batch_images, self.y:batch_labels})
        print("accuracy:", result)

        np.random.seed()
        si = np.random.choice(self.batch_size, len_discrete_code_ply)

        samples = samples[si, :, :, :]
        batch_images = batch_images[si, :, :, :]
        batch_sounds = batch_sounds[si, :, :, :]

        all_samples = np.concatenate((batch_images, batch_sounds), axis=0)
        all_samples = np.concatenate((all_samples, samples), axis=0)

        """ save merged images to check style-consistency """

        canvas = np.zeros_like(all_samples)
        for s in range(3):
            for c in range(len_discrete_code_ply):
                canvas[:, :, :, :] = all_samples[:, :, :, :]
                # canvas[s * self.len_discrete_code + c, :, :, :] = all_samples[c * n_styles + s, :, :, :]

        save_images(canvas, [3, len_discrete_code_ply],
                    check_folder(self.result_dir + '/' + self.model_dir) + '/' + self.model_name +
                    '_epoch%03d' % epoch + '_test_all_classes_style_by_style.png')

    def visualize_results_test(self, epoch):
        test_x, test_sounds, test_sounds_mis, test_y = load_Sub_test(self.dataset_name, self.y_dim)

        z_sample = np.random.normal(0, 1, [self.batch_size, self.z_dim]).astype(np.float32)

        batch_labels = test_y[10 * self.batch_size:(10 + 1) * self.batch_size]
        batch_files = test_x[10 * self.batch_size:(10 + 1) * self.batch_size]
        batch = [
            get_image(batch_file,
                      input_height=self.input_height,
                      input_width=self.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      ) for batch_file in batch_files]

        batch_images = np.array(batch).astype(np.float32)

        batch_sounds_files = test_sounds[10 * self.batch_size:(10 + 1) * self.batch_size]

        batch_S = [
            get_image(batch_file_s,
                      input_height=0,
                      input_width=0,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      crop=False
                      ) for batch_file_s in batch_sounds_files]

        batch_sounds = np.array(batch_S).astype(np.float32)
        batch_sounds = batch_sounds[:, :, :, :3]

        len_discrete_code_ply = 10

        prediction = self.sess.run(self.prediction, feed_dict={self.inputs_img: batch_images})

        samples = self.sess.run(self.fake_images, feed_dict={self.z: z_sample, self.inputs_img: batch_images})

        correct_prediction = np.equal(np.argmax(prediction, 1), np.argmax(batch_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = self.sess.run(accuracy,
                               feed_dict={self.z: z_sample, self.inputs_img: batch_images, self.y: batch_labels})
        print("Image Classifier accuracy:", result)

        np.random.seed()
        si = np.random.choice(self.batch_size, len_discrete_code_ply)

        samples = samples[si, :, :, :]
        batch_images = batch_images[si, :, :, :]
        batch_sounds = batch_sounds[si, :, :, :]

        all_samples = np.concatenate((batch_images, batch_sounds), axis=0)
        all_samples = np.concatenate((all_samples, samples), axis=0)

        """ save merged images to check style-consistency """

        canvas = np.zeros_like(all_samples)
        for s in range(3):
            for c in range(len_discrete_code_ply):
                canvas[:, :, :, :] = all_samples[:, :, :, :]
                # canvas[s * self.len_discrete_code + c, :, :, :] = all_samples[c * n_styles + s, :, :, :]

        save_images(canvas, [3, len_discrete_code_ply],
                    check_folder(self.test_dir + '/' + self.model_dir) + '/' + self.model_name +
                    '_epoch%03d' % epoch + '_test_all_classes_style_by_style.png')

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.model_name, self.dataset_name,
            self.batch_size, self.z_dim)

    def save(self, checkpoint_dir, step):
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name + '.model'), global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir, self.model_name)

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

    def train_check(self):
        import re
        checkpoint_dir = os.path.join(self.checkpoint_dir, self.model_dir, self.model_name)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
            start_epoch = (int)(counter / self.num_batches)
        if start_epoch == self.epoch:
            print(" [*] Training already finished! Begin to test your model")

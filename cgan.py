import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import tensorflow as tf 
from datamanager import datamanager
from networks import Generator, Discriminator
from utils import *
import os

class CGAN(BasicTrainFramework):
    def __init__(self, 
                 batch_size,
                 noise_dim=100, 
                 version="CGAN"):
        super(CGAN, self).__init__(batch_size, version)

        self.noise_dim = noise_dim
        
        self.data = datamanager('CT', train_ratio=0.8, expand_dim=3, seed=0)
        self.data_test = self.data(self.batch_size, 'test', var_list=['data', 'labels'])
        self.class_num = self.data.class_num

        self.Generator = Generator(output_dim=1, name='G')
        self.Discriminator = Discriminator(name='D')

        self.build_placeholder()
        self.build_gan()
        self.build_optimizer()
        self.build_summary()

        self.build_sess()
        self.build_dirs()
    
    def build_placeholder(self):
        self.noise = tf.placeholder(shape=(self.batch_size, self.noise_dim), dtype=tf.float32)
        self.source = tf.placeholder(shape=(self.batch_size, 182, 2, 1), dtype=tf.float32)
        self.labels = tf.placeholder(shape=(self.batch_size, self.class_num), dtype=tf.float32)

    def build_gan(self):
        self.gen = self.Generator(self.noise, self.labels)
        # self.gen_test = self.Generator(self.noise, self.labels, is_training=False, reuse=True)
        
        self.logit_real, _ = self.Discriminator(self.source, self.labels)
        self.logit_fake, _ = self.Discriminator(self.gen, self.labels, reuse=True)

    def build_optimizer(self):
        self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_real, labels=tf.ones_like(self.logit_real)))
        self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_fake, labels=tf.zeros_like(self.logit_fake)))
        self.D_loss = self.D_loss_real + self.D_loss_fake
        self.G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logit_fake, labels=tf.ones_like(self.logit_fake)))
        
        # optimizers
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.D_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.D_loss, var_list=self.Discriminator.vars)
            self.G_solver = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.5).minimize(self.G_loss, var_list=self.Generator.vars)
    
    def build_summary(self):
        D_sum = tf.summary.scalar("D_loss", self.D_loss)
        D_sum_real = tf.summary.scalar("D_loss_real", self.D_loss_real)
        D_sum_fake = tf.summary.scalar("D_loss_fake", self.D_loss_fake)
        G_sum = tf.summary.scalar("G_loss", self.G_loss)
        self.summary = tf.summary.merge([D_sum, D_sum_real, D_sum_fake, G_sum])
    

    def sample(self, epoch=0):
        symbols = ['a','b','c','d','e','g','h','l','m','n','o','p','q','r','s','u','v','w','y','z']

        labels = [[i]*4 for i in range(20)] # no.0 ~ no.79
        labels = np.concatenate(2 * labels) # 160

        # no.0 ~ no.63
        feed_dict = {
            self.labels: one_hot_encode(labels, 20)[:self.batch_size,:],
            self.noise: np.random.uniform(size=(self.batch_size, self.noise_dim), low=-1.0, high=1.0)
        }
        gen_1 = self.sess.run(self.gen, feed_dict=feed_dict)
        # no.64 ~ no.127
        feed_dict = {
            self.labels: one_hot_encode(labels, 20)[self.batch_size:2*self.batch_size,:],
            self.noise: np.random.uniform(size=(self.batch_size, self.noise_dim), low=-1.0, high=1.0)
        }
        gen_2 = self.sess.run(self.gen, feed_dict=feed_dict)
        gen = np.concatenate([gen_1, gen_2], 0)

        for i in range(8):
            for j in range(8):
                plt.subplot(8,8,i*8+j+1)
                plt.plot(gen[i*8+j, :, 0, 0], gen[i*8+j, :, 1, 0], linewidth=2)
                plt.title(symbols[labels[i*8+j]])
                plt.xticks([])
                plt.yticks([])
        plt.savefig(os.path.join(self.fig_dir, 'epoch{}_part1.png'.format(epoch)))
        plt.clf()

        for i in range(8):
            for j in range(8):
                if i*8+j>=16:
                    break
                plt.subplot(8,8,i*8+j+1)
                plt.plot(gen[i*8+j+64, :, 0, 0], gen[i*8+j+64, :, 1, 0], linewidth=2)
                plt.title(symbols[labels[i*8+j+64]])
                plt.xticks([])
                plt.yticks([])
        plt.savefig(os.path.join(self.fig_dir, 'epoch{}_part2.png'.format(epoch)))
        plt.clf()

    def plot_loss(self):
        event = event_reader(self.log_dir,
                names=['D_loss', 'G_loss', 'D_loss_real', 'D_loss_fake'])

        for k,v in event.iteritems():
            print k, len(v)

        plt.clf()
        for i in range(4):
            key = event.keys()[i]
            ax = plt.subplot(2,2,i+1)
             # plt.gca().set_ylim([0, 1])
            plt.plot(range(len(event[key])), event[key])
            plt.xticks(range(0, len(event[key])+1, 350))
            if i>1:
                plt.xlabel("iterx10")
            plt.title(key)

        plt.savefig(os.path.join(self.fig_dir, "loss.png"))
        plt.clf()


    def train(self, epoches=1):
        self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)
        batches_per_epoch = self.data.train_num // self.batch_size
        self.sample(0)

        for epoch in range(epoches):
            self.data.shuffle_train(seed=epoch)

            for idx in range(batches_per_epoch):
                cnt = epoch * batches_per_epoch + idx 

                data = self.data(self.batch_size, var_list=["data", "labels"])
                
                feed_dict = {
                    self.source: data['data'],
                    self.labels: data['labels'],
                    self.noise: np.random.uniform(size=(self.batch_size, self.noise_dim), low=-1.0, high=1.0)
                }

                # train D
                self.sess.run(self.D_solver, feed_dict=feed_dict)

                # train G
                self.sess.run(self.G_solver, feed_dict=feed_dict)

                if cnt % 10 == 0:
                    d, dr, df, g, sum_str = self.sess.run([self.D_loss, self.D_loss_real, self.D_loss_fake, self.G_loss, self.summary], feed_dict=feed_dict)
                    print self.version + " epoch [%3d/%3d] iter [%3d/%3d] D=%.3f Dr=%.3f Df=%.3f G=%.3f" % (epoch, epoches, idx, batches_per_epoch, d, dr, df, g)
                    self.writer.add_summary(sum_str, cnt)

            if (epoch+1) % 50 == 0:
                self.sample(epoch+1)
        self.saver.save(self.sess, os.path.join(self.model_dir, 'model.ckpt'), global_step=cnt)

        

cgan = CGAN(64)
cgan.train(500)
cgan.plot_loss()
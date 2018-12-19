import os
import scipy.misc
import numpy as np

from model import CWGAN

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1000, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("img_height", 64, "The size of image to use")
flags.DEFINE_integer("img_width", 64, "The size of image to use")

flags.DEFINE_string("checkpoint_dir", "checkpoint_MFPT", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("output_dir", "MFPT_samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
FLAGS = flags.FLAGS


def main(_):

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:
        cwgan = CWGAN(
            sess=sess,
            y_dim=3,
            batch_size=FLAGS.batch_size,
            output_dir=FLAGS.output_dir,
            checkpoint_dir=FLAGS.checkpoint_dir)

        if FLAGS.train:
            cwgan.train(FLAGS)
        else:
            if not cwgan.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")
            cwgan.sample("ball", plot=False, save_np=True, sample_times=8)


if __name__ == '__main__':
    tf.app.run()

import argparse
import os
import time
import tensorflow as tf
import numpy as np
from glob import glob
import datetime
import random
from PIL import Image
import matplotlib.pyplot as plt
from train import model_inputs,generator

def test(sess, input_z, out_channel_dim, epoch, number_of_images, args):
    counter =0
    example_z = np.random.uniform(-1, 1, size=[number_of_images, input_z.get_shape().as_list()[-1]])
    samples = sess.run(generator(input_z, out_channel_dim, False, args), feed_dict={input_z: example_z})
    sample_images = [((sample + 1.0) * 127.5).astype(np.uint8) for sample in samples]
    for index, image in enumerate(sample_images):
        image_array = sample_images[index]
        image = Image.fromarray(image_array)
        name= str(counter).zfill(3)
        image.save( os.path.join(args.output_dir, "sample_"+name+  ".jpg"))
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./model.ckpt')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr_d', type=float, default=0.00004, help='Learning rate disciriminator')
    parser.add_argument('--lr_g', type=float, default=0.0004, help='Learning rate generator')
    parser.add_argument('--image_size', type=int, default=128, help='Image size')
    parser.add_argument('--noise_size', type=int, default=100, help='noise size')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1')
    parser.add_argument('--weight_init_stddev', type=float, default=0.02, help='Weight initialization stdev')
    parser.add_argument('--epsilon', type=float, default=0.00005, help='Epsilon')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--n_channel', type=int, default=3, help='Number of channels')
    parser.add_argument('--n_images2generate', type=int, default=100, help='Number of images to generate')
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    input_images, input_z, lr_G, lr_D = model_inputs([args.image_size, args.image_size], args.noise_size)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, args.model_path)
        test(sess,input_z,args.n_channel,args.n_images2generate)
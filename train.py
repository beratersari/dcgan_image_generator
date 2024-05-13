import os
import time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import glob
import datetime
import random
from PIL import Image
import matplotlib.pyplot as plt
import argparse
def generator(z, output_channel_dim, training, args):
    with tf.variable_scope("generator", reuse=not training):
        # 8x8x1024
        fully_connected = tf.layers.dense(z, 8 * 8 * 1024)
        fully_connected = tf.reshape(fully_connected, (-1, 8, 8, 1024))
        fully_connected = tf.nn.leaky_relu(fully_connected)

        # 8x8x1024 -> 16x16x512
        trans_conv1 = tf.layers.conv2d_transpose(inputs=fully_connected,
                                                 filters=512,
                                                 kernel_size=[5, 5],
                                                 strides=[2, 2],
                                                 padding="SAME",
                                                 kernel_initializer=tf.truncated_normal_initializer(
                                                     stddev=args.weight_init_stddev),
                                                 name="trans_conv1")
        batch_trans_conv1 = tf.layers.batch_normalization(inputs=trans_conv1,
                                                          training=training,
                                                          epsilon=args.epsilon,
                                                          name="batch_trans_conv1")
        trans_conv1_out = tf.nn.leaky_relu(batch_trans_conv1,
                                           name="trans_conv1_out")

        # 16x16x512 -> 32x32x256
        trans_conv2 = tf.layers.conv2d_transpose(inputs=trans_conv1_out,
                                                 filters=256,
                                                 kernel_size=[5, 5],
                                                 strides=[2, 2],
                                                 padding="SAME",
                                                 kernel_initializer=tf.truncated_normal_initializer(
                                                     stddev=args.weight_init_stddev),
                                                 name="trans_conv2")
        batch_trans_conv2 = tf.layers.batch_normalization(inputs=trans_conv2,
                                                          training=training,
                                                          epsilon=args.epsilon,
                                                          name="batch_trans_conv2")
        trans_conv2_out = tf.nn.leaky_relu(batch_trans_conv2,
                                           name="trans_conv2_out")

        # 32x32x256 -> 64x64x128
        trans_conv3 = tf.layers.conv2d_transpose(inputs=trans_conv2_out,
                                                 filters=128,
                                                 kernel_size=[5, 5],
                                                 strides=[2, 2],
                                                 padding="SAME",
                                                 kernel_initializer=tf.truncated_normal_initializer(
                                                     stddev=args.weight_init_stddev),
                                                 name="trans_conv3")
        batch_trans_conv3 = tf.layers.batch_normalization(inputs=trans_conv3,
                                                          training=training,
                                                          epsilon=args.epsilon,
                                                          name="batch_trans_conv3")
        trans_conv3_out = tf.nn.leaky_relu(batch_trans_conv3,
                                           name="trans_conv3_out")

        # 64x64x128 -> 128x128x64
        trans_conv4 = tf.layers.conv2d_transpose(inputs=trans_conv3_out,
                                                 filters=64,
                                                 kernel_size=[5, 5],
                                                 strides=[2, 2],
                                                 padding="SAME",
                                                 kernel_initializer=tf.truncated_normal_initializer(
                                                     stddev=args.weight_init_stddev),
                                                 name="trans_conv4")
        batch_trans_conv4 = tf.layers.batch_normalization(inputs=trans_conv4,
                                                          training=training,
                                                          epsilon=args.epsilon,
                                                          name="batch_trans_conv4")
        trans_conv4_out = tf.nn.leaky_relu(batch_trans_conv4,
                                           name="trans_conv4_out")

        # 128x128x64 -> 128x128x3
        logits = tf.layers.conv2d_transpose(inputs=trans_conv4_out,
                                            filters=3,
                                            kernel_size=[5, 5],
                                            strides=[1, 1],
                                            padding="SAME",
                                            kernel_initializer=tf.truncated_normal_initializer(
                                                stddev=args.weight_init_stddev),
                                            name="logits")
        out = tf.tanh(logits, name="out")
        return out

def discriminator(x, args,reuse):
    with tf.variable_scope("discriminator",  reuse=reuse):

        # 128*128*3 -> 64x64x64
        conv1 = tf.layers.conv2d(inputs=x,
                                 filters=64,
                                 kernel_size=[5,5],
                                 strides=[2,2],
                                 padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=args.weight_init_stddev),
                                 name='conv1')
        batch_norm1 = tf.layers.batch_normalization(conv1,
                                                    training=True,
                                                    epsilon=args.epsilon,
                                                    name='batch_norm1')
        conv1_out = tf.nn.leaky_relu(batch_norm1,
                                     name="conv1_out")

        # 64x64x64-> 32x32x128
        conv2 = tf.layers.conv2d(inputs=conv1_out,
                                 filters=128,
                                 kernel_size=[5, 5],
                                 strides=[2, 2],
                                 padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=args.weight_init_stddev),
                                 name='conv2')
        batch_norm2 = tf.layers.batch_normalization(conv2,
                                                    training=True,
                                                    epsilon=args.epsilon,
                                                    name='batch_norm2')
        conv2_out = tf.nn.leaky_relu(batch_norm2,
                                     name="conv2_out")

        # 32x32x128 -> 16x16x256
        conv3 = tf.layers.conv2d(inputs=conv2_out,
                                 filters=256,
                                 kernel_size=[5, 5],
                                 strides=[2, 2],
                                 padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=args.weight_init_stddev),
                                 name='conv3')
        batch_norm3 = tf.layers.batch_normalization(conv3,
                                                    training=True,
                                                    epsilon=args.epsilon,
                                                    name='batch_norm3')
        conv3_out = tf.nn.leaky_relu(batch_norm3,
                                     name="conv3_out")

        # 16x16x256 -> 16x16x512
        conv4 = tf.layers.conv2d(inputs=conv3_out,
                                 filters=512,
                                 kernel_size=[5, 5],
                                 strides=[1, 1],
                                 padding="SAME",
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=args.weight_init_stddev),
                                 name='conv4')
        batch_norm4 = tf.layers.batch_normalization(conv4,
                                                    training=True,
                                                    epsilon=args.epsilon,
                                                    name='batch_norm4')
        conv4_out = tf.nn.leaky_relu(batch_norm4,
                                     name="conv4_out")

        # 16x16x512 -> 8x8x1024
        conv5 = tf.layers.conv2d(inputs=conv4_out,
                                filters=1024,
                                kernel_size=[5, 5],
                                strides=[2, 2],
                                padding="SAME",
                                kernel_initializer=tf.truncated_normal_initializer(stddev=args.weight_init_stddev),
                                name='conv5')
        batch_norm5 = tf.layers.batch_normalization(conv5,
                                                    training=True,
                                                    epsilon=args.epsilon,
                                                    name='batch_norm5')
        conv5_out = tf.nn.leaky_relu(batch_norm5,
                                     name="conv5_out")

        flatten = tf.reshape(conv5_out, (-1, 8*8*1024))
        logits = tf.layers.dense(inputs=flatten,
                                 units=1,
                                 activation=None)
        out = tf.sigmoid(logits)
        return out, logits
def model_loss(input_real, input_z, output_channel_dim, args):
    g_model = generator(input_z, output_channel_dim, True,args)

    noisy_input_real = input_real + tf.random_normal(shape=tf.shape(input_real),
                                                     mean=0.0,
                                                     stddev=random.uniform(0.0, 0.1),
                                                     dtype=tf.float32)

    d_model_real, d_logits_real = discriminator(noisy_input_real, args, reuse=False)
    d_model_fake, d_logits_fake = discriminator(g_model, args,reuse=True)

    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real,
                                                                         labels=tf.ones_like(d_model_real)*random.uniform(0.9, 1.0)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                         labels=tf.zeros_like(d_model_fake)))
    d_loss = tf.reduce_mean(0.5 * (d_loss_real + d_loss_fake))
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake,
                                                                    labels=tf.ones_like(d_model_fake)))
    return d_loss, g_loss

def model_optimizers(d_loss, g_loss,args):
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if var.name.startswith("generator")]
    d_vars = [var for var in t_vars if var.name.startswith("discriminator")]

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    gen_updates = [op for op in update_ops if op.name.startswith('generator')]

    with tf.control_dependencies(gen_updates):
        d_train_opt = tf.train.AdamOptimizer(learning_rate=args.lr_d, beta1=args.beta1).minimize(d_loss, var_list=d_vars)
        g_train_opt = tf.train.AdamOptimizer(learning_rate=args.lr_g, beta1=args.beta1).minimize(g_loss, var_list=g_vars)
    return d_train_opt, g_train_opt

def model_inputs(real_dim, z_dim):
    inputs_real = tf.placeholder(tf.float32, (None, *real_dim), name='inputs_real')
    inputs_z = tf.placeholder(tf.float32, (None, z_dim), name="input_z")
    learning_rate_G = tf.placeholder(tf.float32, name="lr_g")
    learning_rate_D = tf.placeholder(tf.float32, name="lr_d")
    return inputs_real, inputs_z, learning_rate_G, learning_rate_D
def get_batches(data,args):
    batches = []
    for i in range(int(data.shape[0]//args.batch_size)):
        batch = data[i * args.batch_size:(i + 1) * args.batch_size]
        augmented_images = []
        for img in batch:
            image = Image.fromarray(img)
            if random.choice([True, False]):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            augmented_images.append(np.asarray(image))
        batch = np.asarray(augmented_images)
        normalized_batch = (batch / 127.5) - 1.0
        batches.append(normalized_batch)
    return batches
def summarize_epoch(epoch, duration, sess, d_losses, g_losses, input_z, data_shape,args):
    minibatch_size = int(data_shape[0]//args.batch_size)
    print("Epoch {}/{}".format(epoch, args.epochs),
          "\nDuration: {:.5f}".format(duration),
          "\nD Loss: {:.5f}".format(np.mean(d_losses[-minibatch_size:])),
          "\nG Loss: {:.5f}".format(np.mean(g_losses[-minibatch_size:])))
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
        counter+=1
def train_helper(get_batches, data_shape, args, checkpoint_to_load=None):
    input_images, input_z, lr_G, lr_D = model_inputs(data_shape[1:], args.noise_size)
    d_loss, g_loss = model_loss(input_images, input_z, data_shape[3],args)
    d_opt, g_opt = model_optimizers(d_loss, g_loss,args)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        epoch = 0
        iteration = 0
        d_losses = []
        g_losses = []

        for epoch in range(args.epochs):
            epoch += 1
            start_time = time.time()

            for batch_images in get_batches:
                iteration += 1
                batch_z = np.random.uniform(-1, 1, size=(args.batch_size, args.noise_size))
                _ = sess.run(d_opt, feed_dict={input_images: batch_images, input_z: batch_z, lr_D: args.lr_d})
                _ = sess.run(g_opt, feed_dict={input_images: batch_images, input_z: batch_z, lr_G: args.lr_g})
                d_losses.append(d_loss.eval({input_z: batch_z, input_images: batch_images}))
                g_losses.append(g_loss.eval({input_z: batch_z}))

            summarize_epoch(epoch, time.time()-start_time, sess, d_losses, g_losses, input_z, data_shape,args)
        test(sess,input_z,args.n_channel,args.epochs,args.n_images2generate,args)

def train(args):
    # Load images and resize
    input_images = np.asarray(
        [np.asarray(Image.open(file).resize((args.image_size, args.image_size))) for file in glob.glob(args.data_dir + '*')])

    with tf.Graph().as_default():
        train_helper(get_batches(input_images,args), input_images.shape,args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr_d', type=float, default=0.00004, help='Learning rate disciriminator')
    parser.add_argument('--lr_g', type=float, default=0.0004, help='Learning rate generator')
    parser.add_argument('--image_size', type=int, default=128, help='Image size')
    parser.add_argument('--noise_size', type=int, default=100, help='noise size')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1')
    parser.add_argument('--weight_init_stddev', type=float, default=0.02, help='Weight initialization stdev')
    parser.add_argument('--epsilon', type=float, default=0.00005, help='Epsilon')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--n_channel', type=int, default=3, help='Number of channels')
    parser.add_argument('--n_images2generate', type=int, default=50000, help='Number of images to generate')
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Output directory created: \"{args.output_dir}\"")

    train(args)




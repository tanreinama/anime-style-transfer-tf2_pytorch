import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
from PIL import Image

def Generator():
    inputs = layers.Input(shape=[256,256,3])
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    x = inputs
    x = layers.Conv2D(64, 7, strides=1, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(128, 5, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
    x = layers.ReLU()(x)

    s = x
    x = layers.Conv2D(128, 3, strides=1, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, strides=1, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(128, 3, strides=1, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
    x = layers.ReLU()(x)
    x += s

    x = layers.UpSampling2D((2,2))(x)
    x = layers.Conv2D(64, 5, strides=1, padding='same', kernel_initializer=initializer,use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(3, 7, strides=1, padding='same', kernel_initializer=initializer, use_bias=False, activation='tanh')(x)

    x = tf.clip_by_value(inputs + x, -1.0, 1.0)

    return keras.Model(inputs=inputs, outputs=x)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='', help='input image file.')
    parser.add_argument('--output_file', type=str, default='', help='result image file to write.')
    parser.add_argument('--gpu', default=0, type=int, help='visible gpu number.')
    args = parser.parse_args()

    if args.gpu >= 0:
        device = "/GPU:%d"%args.gpu
    else:
        device = "/CPU"

    with tf.device(device):
        model = Generator()

        model.load_weights("Anime.h5")

        im = Image.open(args.input_file)

        a = (np.array(im.resize((256,256)), dtype=np.float32) / 127.5) - 1.
        t = a.reshape((1,256,256,3))

        p = model.predict(t)
        p = p.reshape((256,256,3))
        p = ((p*0.5 + 0.5) * 255).astype(np.uint8)

    d = Image.fromarray(p)
    d.resize((im.width,im.height))
    d.save(args.output_file)


if __name__ == '__main__':
    main()

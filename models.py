import numpy as np
import tensorflow as tf

def generator(wav, avi):
    with tf.compat.v1.variable_scope('generator') as vs:
        current_wav = extract_wav(wav) 
        latent_avi, latents = identity_encoder(avi)
        latent_wav = context_encoder(current_wav)
        cells = tf.compat.v1.keras.layers.GRU(10, return_sequences=True, return_state=True)
        noise, _ = cells(tf.random.normal([tf.shape(latent_wav)[0], latent_wav.shape[1], 1]))
        latent_avi = stacking_to_sequence(latent_avi)
        latent_wav_rnn = tf.concat([latent_wav, noise, latent_avi], -1)
        z = frame_decoder(latent_wav_rnn, latents)
        var = tf.contrib.framework.get_variables(vs)
    return z, var

def identity_encoder(avi, ch=50):
    with tf.compat.v1.variable_scope('identity_encoder'):
        z = avi[:, :, :, :, 0]  # [N,H,W,C]
        z = tf.transpose(z, [0, 3, 1, 2])
        latents = list()
        latents.append(z)
        z = tf.contrib.slim.conv2d(z, ch, 3, 2, activation_fn=None, data_format='NCHW')
        for i in range(4):
            z = tf.nn.relu(tf.contrib.slim.batch_norm(z))
            latents.append(z)
            z = tf.contrib.slim.conv2d(z, ch, 3, 2, activation_fn=None, data_format='NCHW')
        z = tf.contrib.slim.flatten(z)
    return z, latents

def context_encoder(wav, i=0):
    with tf.compat.v1.variable_scope('context_encoder_{}'.format(i)):
        z = tf.transpose(wav, [0, 2, 1])
        z = tf.contrib.slim.conv1d(z, 20, 1, activation_fn=None, data_format='NCW')
        z = tf.transpose(z, [0, 2, 1])
        cells = tf.compat.v1.keras.layers.GRU(256, return_sequences=True, return_state=True)
        z, _ = cells(z)
    return zs

def frame_decoder(wav, latents):
    with tf.compat.v1.variable_scope('frame_decoder'):
        z = tf.expand_dims(tf.expand_dims(wav, -1), -1)
        z = tf.transpose(z, [0, 2, 1, 3, 4])
        for i in range(5):
            z = tf.contrib.slim.conv3d_transpose(z, 50, [1, 4, 4], [1, 2, 2], activation_fn=None, data_format='NCDHW')
            z = tf.nn.relu(tf.contrib.slim.batch_norm(z))
            z = tf.concat([z, stacking_to_sequence_v2(latents[4 - i])], 1)
        z = tf.contrib.slim.conv3d(z, 1, [1, 3, 3], 1, activation_fn=tf.nn.tanh, data_format='NCDHW')
        z = tf.transpose(z, [0, 3, 4, 1, 2])  # [NHWCD]
    return z

def frame_discriminator(avi, G):
    with tf.compat.v1.variable_scope('frame_discriminator') as vs:
        random_value = tf.random_uniform([], minval=0, maxval=tf.shape(avi)[-1], dtype=tf.int32)
        random_avi = avi[:, :, :, :, random_value]
        random_G = G[:, :, :, :, random_value]
        x = tf.concat([random_avi, random_G], 0)  # [2N, 64, 64, 3]
        condition = avi[:, :, :, :, 0]
        x = tf.concat([x, tf.concat([condition, condition], 0)], 3)
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.contrib.slim.conv2d(x, 50, 3, 2, activation_fn=None, data_format='NCHW')
        for i in range(4):
            x = tf.nn.leaky_relu(tf.contrib.slim.batch_norm(x))
            x = tf.contrib.slim.conv2d(x, 50, 3, 2, activation_fn=None, data_format='NCHW')
        x = tf.squeeze(x, [-2, -1])

        var = tf.contrib.framework.get_variables(vs)
    return x, var

def sequence_discriminator(wav, avi, G):
    with tf.compat.v1.variable_scope('sequence_discriminator') as vs:
        wav = extract_wav(wav)  # [B, 75, 20]
        wav = context_encoder(wav)
        # avi = shrink_avi(avi)
        x = tf.concat([avi, G], 0)  # [N,H,W,C,D]
        x = tf.transpose(x, [0, 3, 4, 1, 2])
        x = tf.contrib.slim.conv3d(x, 50, [1, 3, 3], [1, 2, 2], activation_fn=None, data_format='NCDHW')
        for i in range(4):
            x = tf.nn.leaky_relu(tf.contrib.slim.batch_norm(x))
            x = tf.contrib.slim.conv3d(x, 50, [1, 3, 3], [1, 2, 2], activation_fn=None, data_format='NCDHW')
        x = tf.squeeze(x, [-2, -1])
        x = tf.transpose(x, [0, 2, 1])
        x = context_encoder(x, 1)
        x = tf.concat([x, wav], 2)
        x = tf.transpose(x, [0, 2, 1])
        x = tf.nn.leaky_relu(tf.contrib.slim.batch_norm(x))
        x = tf.contrib.slim.conv1d(x, 50, 1, activation_fn=None, data_format='NCW')
        x = tf.nn.leaky_relu(tf.contrib.slim.batch_norm(x))
        x = tf.contrib.slim.conv1d(x, 1, 1, activation_fn=None, data_format='NCW')
        x = tf.squeeze(x, [-2])

        var = tf.contrib.framework.get_variables(vs)
    return x, var
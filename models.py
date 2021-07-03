import tensorflow as tf

OUTPUT_CHANNELS = 1

def downsample(filters, size, strides=(1, 2, 2), apply_batchnorm=True):
    # initializer = tf.random_normal_initializer(0., 0.02)
    initializer = tf.initializers.glorot_uniform()

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv3D(filters, size, strides=strides, padding='same',
                                kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


def upsample(filters, size, strides=(1, 2, 2), apply_dropout=False):
    # initializer = tf.random_normal_initializer(0., 0.02)
    initializer = tf.initializers.glorot_uniform()

    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv3DTranspose(filters, size, strides=strides,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.25))

    result.add(tf.keras.layers.ReLU())

    return result


def Generator():
    inputs = tf.keras.layers.Input(shape=[16, 256, 256, 3])

    down_stack = [
        downsample(32, 4, strides=(2, 2, 2), apply_batchnorm=False),  # (bs, 8, 128, 128, 64)
        downsample(32, 4),  # (bs, 8, 64, 64, 128)
        downsample(64, 4, strides=(2, 2, 2)),  # (bs, 4, 32, 32, 256)
        downsample(128, 4),  # (bs, 4, 16, 16, 512)
        downsample(128, 4),  # (bs, 4, 16, 16, 512)
        downsample(256, 4),  # (bs, 4,  8, 8, 512)
    ]

    up_stack = [
        upsample(256, 4, apply_dropout=True),  # (bs, 4,  8, 8, 1024)
        upsample(128, 4, apply_dropout=True),  # (bs, 4,  16, 16, 1024)
        upsample(128, 4),  # (bs, 4,  16, 16, 1024)
        upsample(64, 4, strides=(2, 2, 2)),  # (bs, 8, 32, 32, 512)
        upsample(32, 4),  # (bs, 8, 64, 64, 256)
        upsample(32, 4),  # (bs, 8, 128, 128, 128)
    ]

    initializer = tf.initializers.glorot_uniform()
    last = tf.keras.layers.Conv3DTranspose(OUTPUT_CHANNELS, 4,
                                            strides=(2, 2, 2),
                                            padding='same',
                                            kernel_initializer=initializer,
                                            activation='tanh')  # (bs, 16, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

LAMBDA = 10

@tf.function
def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def Discriminator():
    initializer = tf.initializers.glorot_uniform()

    inp = tf.keras.layers.Input(shape=[16, 256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[16, 256, 256, 1], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 32, 256, 256, channels*2)

    down1 = downsample(32, 4, apply_batchnorm=False)(x)  # (bs, 32, 128, 128, 64)
    down2 = downsample(64, 4)(down1)  # (bs, 32, 64, 64, 128)
    down3 = downsample(128, 4)(down2)  # (bs, 32, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding3D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv3D(256, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding3D()(leaky_relu)  # (bs, 33, 33, 512)
    
    drop = tf.keras.layers.Dropout(0.5)(zero_pad2)

    last = tf.keras.layers.Conv3D(16, 4, strides=1,
                                kernel_initializer=initializer)(drop)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


@tf.function
def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss
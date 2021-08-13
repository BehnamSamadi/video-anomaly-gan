import tensorflow as tf



OUTPUT_CHANNELS = 1

def downsample(filters, size, strides=(1, 2, 2), apply_batchnorm=True):
    # initializer = tf.random_normal_initializer(0., 0.02)
    initializer = tf.initializers.glorot_uniform()

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv3D(filters, size, strides=strides, padding='same',
                                kernel_initializer=initializer, use_bias=True,
                                kernel_regularizer=tf.keras.regularizers.l1_l2(0.00001, 0.00001)))

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
                                    kernel_initializer=initializer, use_bias=True,
                                    kernel_regularizer=tf.keras.regularizers.l1_l2(0.0001, 0.0001)))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.LeakyReLU())

    return result


def downsample_3d(filters, size, strides=(2, 2, 2), apply_batchnorm=True):
    initializer = tf.initializers.glorot_uniform()

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv3D(filters, size, strides=strides, padding='same',
                                kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsample_3d(filters, size, strides=(2, 2, 2), apply_dropout=False):
    initializer = tf.initializers.glorot_uniform()

    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv3DTranspose(filters, size, strides=strides,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def upsample_2d(filters, size, strides=2, apply_dropout=False):
    initializer = tf.initializers.glorot_uniform()

    result = tf.keras.Sequential()
    result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=strides,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=True))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def downsample_2d(filters, size, strides=2, apply_batchnorm=True):
    initializer = tf.initializers.glorot_uniform()

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',
                                kernel_initializer=initializer, use_bias=True))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

def background():
    inputs = tf.keras.layers.Input(shape=[1, 1, 64])
    conv_1 = upsample_2d(128, 4, strides=4)(inputs)
    conv_2 = upsample_2d(256, 4, strides=4)(conv_1)
    conv_3 = upsample_2d(512, 4, strides=4)(conv_2)
    conv_4 = upsample_2d(256, 4, strides=2)(conv_3)
    conv_4_3d = tf.reshape(conv_4, (-1, 8, 128, 128, 32))
    conv_5 = upsample_3d(OUTPUT_CHANNELS, 4, strides=2)(conv_4_3d)
    # conv_6 = upsample_3d(1, 4, strides=(2, 1, 1))(conv_5)
    out = conv_5
    
    
    return tf.keras.Model(inputs=inputs, outputs=out)


def mask():
    initializer = tf.initializers.glorot_uniform()

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv3D(4, 4, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=False))

    result.add(tf.keras.layers.LeakyReLU())

    result.add(
        tf.keras.layers.Conv3D(8, 4, strides=(4, 2, 2), padding='same',
                                kernel_initializer=initializer, use_bias=True))

    result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())
    
    result.add(
        tf.keras.layers.Conv3DTranspose(8, 4, strides=(4, 2, 2), padding='same',
                                kernel_initializer=initializer, use_bias=True))

    result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    result.add(
        tf.keras.layers.Conv3DTranspose(OUTPUT_CHANNELS, 4, strides=2, padding='same',
                                kernel_initializer=initializer, use_bias=True))

    result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.Activation('sigmoid'))

    return result


def Generator(with_background=True):
    inputs = tf.keras.layers.Input(shape=[16, 256, 256, 3])

    down_stack = [
        downsample(16, 4, strides=(1, 2, 2), apply_batchnorm=False),  # (bs, 8, 128, 128, 64)
        downsample(32, 4, strides=(2, 2, 2)),  # (bs, 8, 64, 64, 128)
        downsample(32, 4, strides=(2, 2, 2)),  # (bs, 8, 64, 64, 128)
        downsample(64, 4, strides=(2, 2, 2)),  # (bs, 4, 32, 32, 256)
        downsample(64, 4, strides=(2, 2, 2)),  # (bs, 4, 16, 16, 512)
        downsample(64, 4, strides=(1, 2, 2)),  # (bs, 4, 16, 16, 512)
        downsample(128, 4, strides=(1, 2, 2)),  # (bs, 4, 16, 16, 512)
    ]

    up_stack = [
        upsample(128, 4, strides=(1, 2, 2), apply_dropout=True),  # (bs, 4,  8, 8, 1024)
        upsample(64, 4, strides=(1, 2, 2), apply_dropout=True),  # (bs, 4,  8, 8, 1024)
        upsample(64, 4, strides=(2, 2, 2), apply_dropout=True),  # (bs, 4,  8, 8, 1024)
        upsample(64, 4, strides=(2, 2, 2)),  # (bs, 4,  16, 16, 1024)
        upsample(32, 4, strides=(2, 2, 2)),  # (bs, 4,  16, 16, 1024)
        upsample(32, 4, strides=(2, 2, 2)),  # (bs, 8, 128, 128, 128)
        upsample(16, 4, strides=(2, 2, 2)),  # (bs, 8, 128, 128, 128)
    ]

    initializer = tf.initializers.glorot_uniform()
    last = tf.keras.layers.Conv3DTranspose(OUTPUT_CHANNELS, 4,
                                            strides=(1, 2, 2),
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
    if not with_background:
        return tf.keras.Model(inputs=inputs, outputs=x)
    
    bg_input = tf.random.uniform((1, 1, 1, 64))
    bg = background()(bg_input)
    msk = mask()(inputs)
    
    out = bg*(1-msk) + x * msk

    return tf.keras.Model(inputs=inputs, outputs=out)

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

LAMBDA = 100

@tf.function
def generator_loss(disc_generated_output, gen_output, target):
    # gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    gan_loss = tf.reduce_mean(tf.pow(disc_generated_output - tf.ones_like(disc_generated_output), 2))

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.pow(target - gen_output, 2))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss


def Discriminator():
    initializer = tf.initializers.glorot_uniform()

    inp = tf.keras.layers.Input(shape=[16, 256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[16, 256, 256, 1], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (bs, 32, 256, 256, channels*2)

    down1 = downsample(32, 4, apply_batchnorm=False)(x)  # (bs, 32, 128, 128, 64)
    down2 = downsample(32, 4, strides=(2, 2, 2))(down1)  # (bs, 32, 64, 64, 128)
    down3 = downsample(64, 4, strides=(2, 2, 2))(down2)  # (bs, 32, 32, 32, 256)
    down4 = downsample(64, 4, strides=(2, 2, 2))(down3)  # (bs, 32, 32, 32, 256)
    down5 = downsample(64, 4, strides=(2, 2, 2))(down4)  # (bs, 32, 32, 32, 256)
    
    up1 = upsample(64, 4, strides=2)(down5)
    up2 = upsample(32, 4, strides=(2, 1, 1))(up1)
    up3 = upsample(32, 4, strides=(2, 1, 1))(up2)
    
    up4 = tf.keras.layers.Conv3DTranspose(1, 4, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=True)(up3)
        
    out = tf.keras.activations.sigmoid(up4)

    return tf.keras.Model(inputs=[inp, tar], outputs=out)


@tf.function
def discriminator_loss(disc_real_output, disc_generated_output):
    # real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    real_loss = tf.reduce_mean(tf.pow(disc_real_output - tf.ones_like(disc_real_output), 2))

    # generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    generated_loss = tf.reduce_mean(tf.pow(disc_generated_output, 2))
    

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss, real_loss, generated_loss
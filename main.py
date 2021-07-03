import os
from numpy.lib.type_check import real
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
import time
from matplotlib import pyplot as plt
from IPython import display
import cv2
from models import *



BUFFER_SIZE = 32
BATCH_SIZE = 4
IMG_WIDTH = 256
IMG_HEIGHT = 256

# tf.debugging.set_log_device_placement(True)
# mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0","/gpu:1"])

# mirrored_strategy.scope()

def load(image_file):
    image = cv2.imread(image_file)
    image = image[:,:,::-1]

    w = image.shape[1]

    w = w // 2
    real_image = image[:, :w, :]
    real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY).astype('float32')
    real_image = np.expand_dims(real_image, axis=-1)
    input_image = image[:, w:, :].astype('float32')
    return input_image, real_image
    


# def load(image_file):

#     image = tf.io.read_file(image_file)
#     image = tf.image.decode_jpeg(image)

#     w = tf.shape(image)[1]

#     w = w // 2
#     real_image = image[:, :w, :]
#     input_image = image[:, w:, :]

#     input_image = tf.cast(input_image, tf.float32)
#     real_image = tf.cast(real_image, tf.float32)

#     return input_image, real_image

def resize(input_image, real_image, height, width):
    input_image = cv2.resize(input_image, (width, height))
    real_image = cv2.resize(real_image, (width, height))
    return input_image, real_image

# def resize(input_image, real_image, height, width):
#     input_image = tf.image.resize(input_image, [height, width],
#                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#     real_image = tf.image.resize(real_image, [height, width],
#                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

#     return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

def salt_pepper_noise(input_image, real_image):
    if tf.random.uniform(()) > 0.5:
        SNR = 0.99
        h, w, c = input_image.shape
        mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[SNR, (1 - SNR) / 5., (1 - SNR) * 4 / 5.])
        Mask = np.repeat(mask, c, axis=2) # Copy by channel to have the same shape as img
        input_image[Mask == 1] = 128 # salt noise
        input_image[Mask == 2] = 0
    return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
    # resizing to 286 x 286 x 3
    # input_image, real_image = resize(input_image, real_image, 286, 286)

    # # randomly cropping to 256 x 256 x 3
    # input_image, real_image = random_crop(input_image, real_image)

    # if tf.random.uniform(()) > 0.5:
    # # random mirroring
    #     input_image = tf.image.flip_left_right(input_image)
    #     real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image



def load_train_frames(txt_file):
    input_frames = []
    real_frames = []
    txt = tf.io.read_file(txt_file)
    frame_list = txt.numpy().decode('utf-8').split('\n')[:-1]
    for frame in frame_list:
        inp, real = load_image_train(frame)
        input_frames.append(inp)
        real_frames.append(real)
    # input_frames = tf.convert_to_tensor(input_frames, shape=(16, 256, 256, 3))
    # real_frames = tf.convert_to_tensor(real_frames, shape=(16, 256, 256, 3))
    # print(real_frames)
    # saggg
    return input_frames, real_frames

def load_test_frames(txt_file):
    input_frames = []
    real_frames = []
    txt = tf.io.read_file(txt_file)
    frame_list = txt.numpy().decode('utf-8').split('\n')[:-1]
    for frame in frame_list:
        inp, real = load_image_test(frame)
        input_frames.append(inp)
        real_frames.append(real)
    # input_frames = tf.convert_to_tensor(input_frames, shape=(16, 256, 256, 3))
    # real_frames = tf.convert_to_tensor(real_frames, shape=(16, 256, 256, 3))
    
    return input_frames, real_frames



def load_image_train(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = salt_pepper_noise(input_image, real_image)
    input_image, real_image = random_jitter(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                    IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

PATH = '../../ped1/'
PATH = 'dataset/16frames/'

train_dataset = tf.data.Dataset.list_files(PATH+'train/*.txt')
train_dataset = train_dataset.map(lambda x: tf.py_function(load_train_frames, [x], [tf.float32, tf.float32]), num_parallel_calls=4)
                                #num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(1)

# test_dataset = tf.data.Dataset.list_files(PATH+'test/*.txt')
# test_dataset = test_dataset.map(lambda x: tf.py_function(load_test_frames, [x], [tf.float32, tf.float32]))
# test_dataset = test_dataset.batch(BATCH_SIZE)


generator = Generator()
# tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

discriminator = Discriminator()
# tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)


generator_optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(5e-3, beta_1=0.9)


checkpoint_dir = '/home/sensifai/behnam/anomaly/pix2pix/c3d/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator,
                                discriminator=discriminator)
# status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    discriminator.trainable_variables)
        
        # generator_gradients = tf.gradients(gen_total_loss,
        #                                         generator.trainable_variables)
        # discriminator_gradients = tf.gradients(disc_loss,
        #                                             discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_gradients,
                                                generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)
    
    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss


def fit(train_ds, epochs):
    for epoch in range(epochs):
        start = time.time()

        # display.clear_output(wait=True)

        # for example_input, example_target in test_ds.take(1):
        #     generate_images(generator, example_input, example_target)
        # print("Epoch: ", epoch)

        # Train
        # st = time.time()
        for n, (input_image, target) in train_ds.enumerate():
            # print('load:', time.time()-st)
            print('.', end='')
            if (n+1) % 10 == 0:
                print('\n')
            # train_step(input_image, target, epoch)
            # st = time.time()
            gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(input_image, target, epoch)
            # print('grad:', time.time()-st)
            if (n+1) % 10 == 0:
                print('gen_total_loss', gen_total_loss, epoch)
                print('gen_gan_loss', gen_gan_loss, epoch)
                print('gen_l1_loss', gen_l1_loss, epoch)
                print('disc_loss', disc_loss, epoch)
                print('step', n)
            # st = time.time()
        print()

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 1 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                            time.time()-start))
    checkpoint.save(file_prefix=checkpoint_prefix)

EPOCHS = 50
fit(train_dataset, EPOCHS)

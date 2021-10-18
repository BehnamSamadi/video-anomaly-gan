import os
from numpy.lib.type_check import real
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np
import time
import cv2
# from models.large import *
from models.medium import *
# from models.light import *
# from models.tiny import *

use_backgroud = False


gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

BUFFER_SIZE = 16
BATCH_SIZE = 4
IMG_WIDTH = 256
IMG_HEIGHT = 256


def load(image_file):
    image = cv2.imread(image_file)
    image = image[:,:,::-1]

    w = image.shape[1]

    w = w // 2
    real_image = image[:, :w, :].astype('float32')
    real_image = cv2.cvtColor(real_image, cv2.COLOR_BGR2GRAY).astype('float32')
    real_image = np.expand_dims(real_image, axis=-1)
    input_image = image[:, w:, :].astype('float32')
    return input_image, real_image
    

def resize(input_image, real_image, height, width):
    input_image = cv2.resize(input_image, (width, height))
    real_image = cv2.resize(real_image, (width, height))
    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


@tf.function()
def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

def salt_pepper_noise(input_image, real_image):
    SNR = 1 - tf.random.uniform(())/50
    h, w, c = input_image.shape
    mask = np.random.choice((0, 1), size=(h, w, 1), p=[SNR, (1 - SNR)])
    Mask = np.repeat(mask, c, axis=2) # Copy by channel to have the same shape as img
    input_image[Mask == 1] = 0
    return input_image, real_image


def load_train_frames(txt_file):
    input_frames = []
    real_frames = []
    txt = tf.io.read_file(txt_file)
    frame_list = txt.numpy().decode('utf-8').split('\n')[:-1]
    for frame in frame_list:
        inp, real = load_image_train(frame)
        # inp, real = salt_pepper_noise(inp, real)
        input_frames.append(inp)
        real_frames.append(real)
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

    return input_frames, real_frames


def load_image_train(image_file):
    input_image, real_image = load(image_file)
    # input_image, real_image = salt_pepper_noise(input_image, real_image)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image

PATH = '../c3d/dataset/32frames_alt2/'

train_dataset = tf.data.Dataset.list_files(PATH+'train/*.txt')
train_dataset = train_dataset.map(lambda x: tf.py_function(load_train_frames, [x], [tf.float32, tf.float32]), num_parallel_calls=4)
                                #num_parallel_calls=tf.data.experimental.AUTOTUNE)

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.prefetch(1)

test_dataset = tf.data.Dataset.list_files(PATH+'test/*.txt')
test_dataset = test_dataset.map(lambda x: tf.py_function(load_test_frames, [x], [tf.float32, tf.float32]))
test_dataset = test_dataset.batch(BATCH_SIZE)

print('datasets created')
generator = Generator(with_background=use_backgroud)
discriminator = Discriminator()


generator_optimizer = tf.keras.optimizers.Adam(5e-4, beta_1=0.9, beta_2=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(5e-4, beta_1=0.9, beta_2=0.5)


checkpoint_dir = './training_checkpoints/medium_ped1'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator,
                                discriminator=discriminator)

status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))



def generate_images(model, test_input, tar, step, use_test):
    if use_backgroud:
        noise_input = tf.random.uniform((test_input.shape[0], 1, 1, 64))
        prediction = model([test_input, noise_input], training=True)
    else:
        prediction = model(test_input, training=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    file_name = 'run_epoch_res/out_c{}_{}.mp4'.format(step, use_test)
    writer = cv2.VideoWriter(file_name, fourcc, 2, (512, 512))

    for inp, t, pred in zip(test_input[0], tar[0], prediction[0]):
        
        inp = (inp+1) * 127.5
        t = np.reshape(t, (256, 256))
        pred = np.reshape(pred, (256, 256))
        diff = np.abs(pred-t) * 255

        t = cv2.cvtColor(t, cv2.COLOR_GRAY2BGR)
        pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)
        diff = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        frame_up = (np.hstack([t, pred]) + 1) * 127.5
        frame_down = np.hstack([inp, diff])
        frame = np.vstack([frame_up, frame_down])
        frame = frame.astype('uint8')
        writer.write(frame)
    writer.release()


import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
    log_dir + "fit/light_ped2/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

@tf.function
def train_step(input_image, target, epoch):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        if use_backgroud:
            noise_input = tf.random.uniform((input_image.shape[0], 1, 1, 64))

            gen_output = generator([input_image, noise_input], training=True)
        else:
            gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss, real_loss, generated_loss = discriminator_loss(disc_real_output, disc_generated_output)

        generator_gradients = gen_tape.gradient(gen_total_loss,
                                                generator.trainable_variables)
        discriminator_gradients = disc_tape.gradient(disc_loss,
                                                    discriminator.trainable_variables)
        
        generator_optimizer.apply_gradients(zip(generator_gradients,
                                                generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                    discriminator.trainable_variables))
    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_total_loss', disc_loss, step=epoch)
        tf.summary.scalar('disc_real_loss', real_loss, step=epoch)
        tf.summary.scalar('disc_gen_loss', generated_loss, step=epoch)
    
    return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss


def fit(train_ds, test_ds, epochs):
    for epoch in range(epochs):
        start = time.time()
        st = start
        for n, (input_image, target) in train_ds.enumerate():
            print('.', end='')
            if (n+1) % 10 == 0:
                print('\n')
            st = time.time()
            gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(input_image, target, epoch)
            if (n+1) % 10 == 0:
                print('gen_total_loss', gen_total_loss, epoch)
                print('gen_gan_loss', gen_gan_loss, epoch)
                print('gen_l1_loss', gen_l1_loss, epoch)
                print('disc_loss', disc_loss, epoch)
                print('step', n)

            if (n+1) % 300 == 0:
                filenameprefix = 'train_{}_{}'.format(epoch, n)
                generate_images(generator, input_image, target, filenameprefix, 0)
                for input_image, target in test_ds.take(1):
                    filenameprefix = 'val_{}_{}'.format(epoch, n)
                    generate_images(generator, input_image, target, filenameprefix, 0)
        print()

        # saving (checkpoint) the model every i epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                            time.time()-start))
    checkpoint.save(file_prefix=checkpoint_prefix)

EPOCHS = 50
fit(train_dataset, test_dataset, EPOCHS)

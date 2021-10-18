import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from numpy.lib.type_check import real
from tensorflow.python.ops.numpy_ops.np_math_ops import diff

import tensorflow as tf
import numpy as np
import time
from tqdm import tqdm
import cv2
import glob as gb
from models.light import *



gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)



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


generator = Generator(with_background=False)
discriminator = Discriminator()


generator_optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.9, beta_2=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.9, beta_2=0.5)


checkpoint_dir = './training_checkpoints/light_ped2'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")


checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator,
                                discriminator=discriminator)

status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))


def generate_output(dataset, window_size):
    frame_count = len(dataset)+window_size
    diffs = np.zeros((frame_count, 256, 256))
    for n, inst in enumerate(tqdm(dataset)):
    # for n in tqdm(range(0, len(dataset), 32)):
    #     inst = dataset[n]
        input_image, target = load_test_frames(inst)
        input_image = tf.reshape(input_image, (1, 16, 256, 256, 3))
        gen_output = generator(input_image, training=True)
        for i, (frame_gen, frame_tar) in enumerate(zip(gen_output[0], target)):
            diff = (frame_gen + 1) * 127.5
            diff = np.reshape(diff, (256, 256)) #/ diff.max()
            
            diffs[n+i] += diff
    return diffs
        


def get_disc_out(dataset, window_size):
    frame_count = len(dataset)+window_size
    diffs = np.zeros((frame_count, 128, 128))
    for n, inst in enumerate(tqdm(dataset)):
        input_image, target = load_test_frames(inst)
        input_image = tf.reshape(input_image, (1, 32, 256, 256, 3))
        target = tf.reshape(target, (1, 32, 256, 256, 1))
        disc_output = discriminator([input_image, target], training=True)
        for i, disc in enumerate(disc_output[0]):
            disc = np.reshape(disc, (128, 128))
            diffs[n+i] += disc
        return diffs




out_model = 1

for i in range(12):
    dataset_path = '../c3d/dataset/ped2_16frames/test/{:03d}_*.txt'.format(i)
    dataset = gb.glob(dataset_path)
    dataset.sort()
    window_size = 16
    print(i)
    if out_model:
        diffs = generate_output(dataset, window_size)
        result_path = './results/gen_out/ped2_16_out_50/{:03d}/'.format(i)
    else:
        diffs = get_disc_out(dataset, window_size)
        result_path = './results/disc_out/ped2_16_out_50/{:03d}/'.format(i)

    if not os.path.exists(result_path):
        os.makedirs(result_path)
    for j, diff in enumerate(diffs):
        div = min(j+1, len(diffs)-j, window_size)
        # div = 1
        diff = (diff/div).astype('uint8')
        save_file_name = result_path + '{:03d}.jpg'.format(j+2)
        cv2.imwrite(save_file_name, diff)



import tensorflow as tf
import os
from models import *
from matplotlib import pyplot as plt
import cv2
import numpy as np
import sys
import time



BATCH_SIZE=1
BUFFER_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256

checkpoint_directory = "training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

# Create a Checkpoint that will manage two objects with trackable state,
# one we name "optimizer" and the other we name "model".

use_test = int(sys.argv[1])
step_name = sys.argv[2]


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

def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
    # # resizing to 286 x 286 x 3
    # input_image, real_image = resize(input_image, real_image, 286, 286)

    # # randomly cropping to 256 x 256 x 3
    # input_image, real_image = random_crop(input_image, real_image)

    # if tf.random.uniform(()) > 0.5:
    # # random mirroring
    #     input_image = tf.image.flip_left_right(input_image)
    #     real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image



def load_train_frames(txt_file):
    st = time.time()
    input_frames = []
    real_frames = []
    txt = tf.io.read_file(txt_file)
    frame_list = txt.numpy().decode('utf-8').split('\n')[:-1]
    print('text time: ', time.time()-st)
    st = time.time()
    for frame in frame_list:
        inp, real = load_image_train(frame)
        input_frames.append(inp)
        real_frames.append(real)
    print('load time: ', time.time()-st)
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
    st = time.time()
    input_image, real_image = load(image_file)
    print('load time: ', time.time()-st)
    st = time.time()
    input_image, real_image = random_jitter(input_image, real_image)
    print('jitter time: ', time.time()-st)
    st = time.time()
    input_image, real_image = normalize(input_image, real_image)
    print('normalize time: ', time.time()-st)
    st = time.time()

    return input_image, real_image


def load_image_test(image_file):
    input_image, real_image = load(image_file)
    input_image, real_image = resize(input_image, real_image,
                                    IMG_HEIGHT, IMG_WIDTH)
    input_image, real_image = normalize(input_image, real_image)

    return input_image, real_image



# PATH = '../../ped1/'
PATH = 'dataset/16frames/'

train_dataset = tf.data.Dataset.list_files(PATH+'train/*.txt')
train_dataset = train_dataset.map(lambda x: tf.py_function(load_train_frames, [x], [tf.float32, tf.float32]))
                                #num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)


test_dataset = tf.data.Dataset.list_files(PATH+'test/*.txt')
test_dataset = test_dataset.map(lambda x: tf.py_function(load_test_frames, [x], [tf.float32, tf.float32]))
test_dataset = test_dataset.batch(BATCH_SIZE)


generator = Generator()
# tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

discriminator = Discriminator()
# tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)


generator_optimizer = tf.keras.optimizers.Adam(2e-6, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-6, beta_1=0.5)


checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                discriminator_optimizer=discriminator_optimizer,
                                generator=generator,
                                discriminator=discriminator)
status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_directory))

# checkpoint_dir = '/home/sensifai/behnam/anomaly/pix2pix/c3d/training_checkpoints'
# checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
# checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                 discriminator_optimizer=discriminator_optimizer,
#                                 generator=generator,
#                                 discriminator=discriminator)



def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    file_name = 'out_c{}_{}.mp4'.format(step_name, use_test)
    writer = cv2.VideoWriter(file_name, fourcc, 2, (512, 512))

    display_list = [test_input[0][0], tar[0][0], prediction[0][0]]
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
        print(frame.dtype)
        # frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        writer.write(frame)
    writer.release()
    

    # title = ['Input Image', 'Ground Truth', 'Predicted Image']

    # for i in range(3):
    #     plt.subplot(1, 3, i+1)
    #     plt.title(title[i])
    #     # getting the pixel values between [0, 1] to plot it.
    #     plt.imshow(display_list[i] * 0.5 + 0.5)
    #     plt.axis('off')
    # plt.show()


# import datetime
# log_dir="logs/"

# summary_writer = tf.summary.create_file_writer(
#     log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# @tf.function
# def train_step(input_image, target, epoch):
#     with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#         gen_output = generator(input_image, training=True)

#         disc_real_output = discriminator([input_image, target], training=True)
#         disc_generated_output = discriminator([input_image, gen_output], training=True)

#         gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
#         disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

#         generator_gradients = gen_tape.gradient(gen_total_loss,
#                                                 generator.trainable_variables)
#         discriminator_gradients = disc_tape.gradient(disc_loss,
#                                                     discriminator.trainable_variables)

#         generator_optimizer.apply_gradients(zip(generator_gradients,
#                                                 generator.trainable_variables))
#         discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
#                                                     discriminator.trainable_variables))

#     with summary_writer.as_default():
#         tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
#         tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
#         tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
#         tf.summary.scalar('disc_loss', disc_loss, step=epoch)
    
#     return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss


# def fit(train_ds, epochs, test_ds):
#     for epoch in range(epochs):
#         start = time.time()

#         # display.clear_output(wait=True)

#         # for example_input, example_target in test_ds.take(1):
#         #     generate_images(generator, example_input, example_target)
#         # print("Epoch: ", epoch)

#         # Train
#         for n, (input_image, target) in train_ds.enumerate():
#             print('.', end='')
#             if (n+1) % 5 == 0:
#                 print('\n')
#             gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(input_image, target, epoch)

#             print('gen_total_loss', gen_total_loss, epoch)
#             print('gen_gan_loss', gen_gan_loss, epoch)
#             print('gen_l1_loss', gen_l1_loss, epoch)
#             print('disc_loss', disc_loss, epoch)
            
#         print()

#         # saving (checkpoint) the model every 20 epochs
#         if (epoch + 1) % 1 == 0:
#             checkpoint.save(file_prefix=checkpoint_prefix)

#         print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
#                                                             time.time()-start))
#     checkpoint.save(file_prefix=checkpoint_prefix)

# EPOCHS = 50
# fit(train_dataset, EPOCHS, test_dataset)

ds = train_dataset
if use_test == 1:
    ds = test_dataset
for example_input, example_target in ds.take(1):
    generate_images(generator, example_input, example_target)

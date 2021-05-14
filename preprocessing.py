import os
import cv2
import glob
import tensorflow as tf
import numpy as np
import librosa
import logging

def get_psnr_mse(image_1, image_2):

    img1 = cv2.imread(image_1)
    img2 = cv2.imread(image_2)
    img2_resize = cv2.resize(img2, (360, 288))

    psnr = cv2.PSNR(img1, img2_resize)
    mse = np.mean((img1 - img2_resize) ** 2)

    return psnr, mse

def video_to_frame():
    path = "./data/s1.mpg_vcd/s1"
    root = glob.glob("{}/*.{}".format(path, "mpg"))
    for i in range(len(root)):
        vidcap = cv2.VideoCapture(root[i])
        success,image = vidcap.read()
        count = 0
        current_name, _ = os.path.splitext(os.path.basename(root[i]))
        file_path = "{}/{}".format(path, current_name)
        if os.path.isdir(file_path) is False:
            os.mkdir(file_path)
        while success:
          cv2.imwrite("{}/{}/{number:05}.jpg".format(path, current_name, number=count),image)
          count += 1

def audio_to_npy():
    path = "./data/s1.mpg_vcd/s1"
    new_path = "./data/s1.mpg_vcd/s1_audio_npy"
    root = glob.glob("{}/*.{}".format(path, "mpg"))
    if os.path.isdir(new_path) is False:
        os.mkdir(new_path)
    for i in range(len(root)):
        current_name, _ = os.path.splitext(os.path.basename(root[i]))
        y, sr = librosa.load(root[i], sr=100)
        np.save("{}/{}.npy".format(new_path, current_name), y)

def load_and_preprocess_video_audio(audio, video):
    audio = tf.numpy_function(read_npy_file, [audio], tf.float32)
    video = tf.numpy_function(read_npy_file, [video], tf.float32)
    audio = tf.reshape(audio, [298])
    audio = tf.concat([audio, [0., 0.]], 0)
    video = tf.reshape(video, [32, 40, 75])
    video, _ = tf.split(video, [32, -1], axis=1)
    video = tf.expand_dims(video, -2)
    return audio, video

def read_npy_file(video):
    file = np.load(video)
    return file.astype(np.float32)

def get_loader(path, batch_size):
    with tf.compat.v1.variable_scope("tfData"):
        audio = glob.glob("{}/*.{}".format(path[0], "npy"))
        video = glob.glob("{}/*.{}".format(path[1], "npy"))
        whole_queue_0 = tf.data.Dataset.from_tensor_slices((audio, video))
        whole_queue = tf.data.Dataset.zip(whole_queue_0)
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        whole_queue = whole_queue.shuffle(buffer_size=1001)
        whole_queue = whole_queue.repeat()
        whole_queue = whole_queue.map(load_and_preprocess_video_audio, num_parallel_calls=AUTOTUNE)
        whole_queue = whole_queue.batch(batch_size)
        whole_queue = whole_queue.prefetch(buffer_size=AUTOTUNE)
    return whole_queue


def prepare_dirs_and_logger(config):
    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    logger = logging.getLogger()

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger.addHandler(handler)

    if config.load_path:
        if config.load_path.startswith(config.log_dir):
            config.model_dir = config.load_path
        else:
            if config.load_path.startswith(config.dataset):
                config.model_name = config.load_path
            else:
                config.model_name = "{}_{}".format(config.dataset, config.load_path)
    else:
        config.model_name = "{}_model".format(config.dataset)

    if not hasattr(config, 'model_dir'):
        config.ckpt_dir = os.path.join(config.check_dir, config.model_name)
    config.data_path = os.path.join(config.data_dir[0], config.dataset)
    config.ckpt_dir = os.path.join(config.check_dir, config.model_name)
    config.model_dir = os.path.join(config.log_dir, config.model_name)

    for path in [config.log_dir, config.data_dir[0], config.model_dir, config.ckpt_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

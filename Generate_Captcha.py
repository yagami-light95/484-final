# -*- coding: utf-8 -*-
from captcha.image import ImageCaptcha
import os
import random
from tqdm import tqdm
from PIL import Image
import numpy as np


class Config(object):
    width = 160  # Set the width of the image
    height = 60  # Set the height of the image
    char_num = 4  # Set the number of CAPTCHA to 4
    characters = range(10)
    generate_num = (10000, 500, 500)  # Number of training sets, validation sets and test sets

    test_folder = 'test'
    train_folder = 'train'
    validation_folder = 'validation'
    tensorboard_folder = 'tensorboard'  # the log path of tensorboard
    saver_folder = 'checkpoints' 

    alpha = 1e-3  # learning
    Epoch = 100  # number of epoch
    batch_size = 64  # batchsize
    keep_prob = 0.5  # dropout
    print_per_batch = 20  # print results every 20 times
    save_per_batch = 20  # Write to tensorboard every 20 times


class Generate:

    def __init__(self):
        self.image = ImageCaptcha(width=Config.width, height=Config.height)
        self.check_path(Config.test_folder)
        self.check_path(Config.validation_folder)
        self.check_path(Config.train_folder)
        self.run()

    @staticmethod
    def check_path(folder):
        # Check if the folder exists and create it if it does not
        if os.path.exists(folder):
            pass
        else:
            os.mkdir(folder)

    def gen_captcha(self, folder, gen_num, random_=True):
        # Generate captcha images
        desc = '{:<10}'.format(folder)

        if random_:
            # Randomly generate captcha for test sets and validation sets
            for _ in tqdm(range(gen_num), desc=desc):
                while True:
                    label = ''.join('%s' % num for num in
                                    random.sample(Config.characters, Config.char_num))
                    path = folder + '/%s.jpg' % label

                    # Check if the captcha already exists
                    if not os.path.exists(path):
                        self.image.generate_image(label)
                        self.image.write(label, path)
                        break

        else:
            # Generate CAPTCHA in order
            for num in tqdm(range(gen_num), desc=desc):
                num_length = len(str(num))

                if num_length < Config.char_num:
                    # Less than 4 digits are made up by 0
                    label = '0' * (Config.char_num - num_length) + str(num)
                    path = folder + '/%s.jpg' % label
                    self.image.generate_image(label)
                    self.image.write(label, path)
                else:
                    label = str(num)
                    path = folder + '/%s.jpg' % label
                    self.image.generate_image(label)
                    self.image.write(label, path)

    def run(self):
        print '==> Generating images...'
        self.gen_captcha(Config.train_folder, Config.generate_num[0], random_=False)
        self.gen_captcha(Config.validation_folder, Config.generate_num[1])
        self.gen_captcha(Config.test_folder, Config.generate_num[2])


class ReadData:

    def __init__(self):
        self.test_img = os.listdir(Config.test_folder)
        self.train_img = os.listdir(Config.train_folder)
        self.sample_num = len(self.train_img)

    def read_data(self, path):
        img = Image.open(path).convert('L')
        image_array = np.array(img)
        image_data = image_array.flatten() / 255.0
        # Cut image path
        label = os.path.splitext(os.path.split(path)[1])[0]
        label_vec = self.label2vec(label)
        return image_data, label_vec

    @staticmethod
    def label2vec(label):
        """
        Convert the CAPTCHA labels to a 40-dimensional vector.
        :param label: 1327
        :return:
            [0,1,0,0,0,0,0,0,0,0,
            0,0,0,1,0,0,0,0,0,0,
            0,0,1,0,0,0,0,0,0,0,
            0,0,0,0,0,0,0,1,0,0]
        """
        label_vec = np.zeros(Config.char_num * len(Config.characters))
        for i, num in enumerate(label):
            idx = i * len(Config.characters) + int(num)
            label_vec[idx] = 1
        return label_vec

    def load_data(self, folder):
        """
        Loading sample data
        :param folder: Folder for pictures
        :return:
            data: image data
            label:  image label
            size:   number of image
        """
        if os.path.exists(folder):
            path_list = os.listdir(folder)
            size = len(path_list)
            data = np.zeros([size, Config.height * Config.width])
            label = np.zeros([size, Config.char_num * len(Config.characters)])
            for i, img_path in enumerate(path_list):
                path = '%s/%s' % (folder, img_path)
                data[i], label[i] = self.read_data(path)
            return data, label, size
        else:
            raise IOError('No such directory, please check again.')


if __name__ == '__main__':
    Generate()

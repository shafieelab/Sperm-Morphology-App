import os
import csv
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import matplotlib.pyplot as plt

from typing import Union, List
import numpy
import numpy as np
import cv2
import os


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def load_image(image: Union[str, numpy.ndarray]) -> numpy.ndarray:
    # Image provided ad string, loading from file ..
    if isinstance(image, str):
        # Checking if the file exist
        if not os.path.isfile(image):
            print("File {} does not exist!".format(image))
            return None
        # Reading image as numpy matrix in gray scale (image, color_param)
        return cv2.imread(image)

    # Image alredy loaded
    elif isinstance(image, numpy.ndarray):
        return image

    # Format not recognized
    else:
        print("Unrecognized format: {}".format(type(image)))
        print("Unrecognized format: {}".format(image))
    return None


def generate_slides_txt_phone(root_dir, run_id):
    # data_dir_path_root = 'data/slides_heads/'
    # data_dir_path_input = data_dir_path_root + ''
    data_dir_path = root_dir + run_id

    output_path = root_dir + '/' + run_id + '/' + run_id + '.txt'
    data_dir_path = data_dir_path + '/slides_heads/'
    slides = [slide for slide in os.listdir(data_dir_path)]

    with open(output_path, 'w') as the_file:

        for slide_name in slides:

            if '.txt' not in slide_name:

                # extra_path_for_sperm_data = '/'

                images_dir = os.listdir(data_dir_path + slide_name)
                # exit()
                # print(slide_name)
                # print( len(images_dir))
                print(slide_name, len(images_dir))

                # random.Random(4).shuffle(images_dir)
                # images_dir = images_dir[0:40]

                # print(len(images_dir))

                # if not len(images_dir) <= 100:
                #
                #     # images_dir = images_dir[int(len(images_dir) * 0): int(len(images_dir) * 0.2)]
                #     print(slide_name,len(images_dir))

                for img_name in images_dir:
                    the_file.write(
                        data_dir_path + slide_name + "/" + img_name + " " + img_name + "," + slide_name + '\n')
        return output_path

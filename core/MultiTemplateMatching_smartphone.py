from os.path import isdir
from MTM import matchTemplates
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import os

# temp_1_1 = cv2.imread("templates/Templates_Smartphone/1_1.jpg")
# temp_1_2 = cv2.imread("Templates_Smartphone/1_2.jpg")
# temp_1_3 = cv2.imread("templates/Templates_Smartphone/1_3.jpg")
# temp_1_4 = cv2.imread("Templates_Smartphone/1_4.jpg")
# temp_1_5 = cv2.imread("templates/Templates_Smartphone/1_5.jpg")
# temp_6 = cv2.imread("Templates_Microscope/6_head.png")
#
# temp_7_1 = cv2.imread("Templates_Microscope/7 (1).png")
# temp_7_2 = cv2.imread("Templates_Microscope/7 (2).png")
# temp_7_3 = cv2.imread("Templates_Microscope/7 (3).png")
#
# temp_8_1 = cv2.imread("Templates_Microscope/8 (1).png")
# temp_8_2 = cv2.imread("Templates_Microscope/8 (2).png")
#
#
# temp_10_1 = cv2.imread("Templates_Microscope/10 (1).png")
# temp_10_2 = cv2.imread("Templates_Microscope/10 (2).png")
# temp_10_3 = cv2.imread("Templates_Microscope/10 (3).png")
# temp_10_4 = cv2.imread("Templates_Microscope/10 (4).png")
# temp_10_5 = cv2.imread("Templates_Microscope/10 (5).png")
#
#
# temp_11_1 = cv2.imread("Templates_Microscope/11 (1).png")
# temp_11_2 = cv2.imread("Templates_Microscope/11 (2).png")
#
# temp_12_1 = cv2.imread("Templates_Microscope/12 (1).png")
# temp_13_1 = cv2.imread("Templates_Microscope/13 (1).png")
# temp_14_1 = cv2.imread("Templates_Microscope/14 (1).png")
# temp_15_1 = cv2.imread("Templates_Microscope/15 (1).png")
# temp_12_2 = cv2.imread("Templates_Microscope/12 (2).png")
#
# template_list = [("temp_1", temp_1), ("temp_2", temp_2), ("temp_3", temp_3), ("temp_4", temp_4), ("temp_5", temp_5),
#                  ("temp_5", temp_6),
#                  ("temp_7_1", temp_7_1),("temp_7_2", temp_7_2),("temp_7_1", temp_7_3),
#                  ("temp_10_1", temp_10_1),("temp_10_2", temp_10_2),("temp_10_1", temp_10_3),("temp_10_2", temp_10_4),("temp_10_1", temp_10_5),
#                  ("temp_11_1", temp_11_1),("temp_11_2", temp_11_2),
#                  ("temp_12_1", temp_12_1),("temp_12_2", temp_12_2),
#                  ("temp_13_1", temp_13_1),
#                  ("temp_14_1", temp_14_1),
#                  ("temp_15_1", temp_15_1),
#
#
#                  ]


temp_sd1_1 = cv2.imread("/var/www/html/Sperm_Morphology_App/templates/human_smartphone_templates/File_1.png")
temp_sd1_2 = cv2.imread("/var/www/html/Sperm_Morphology_App/templates/human_smartphone_templates/File_2.png")
temp_sd1_3 = cv2.imread("/var/www/html/Sperm_Morphology_App/templates/human_smartphone_templates/File_3.png")
temp_sd1_4 = cv2.imread("/var/www/html/Sperm_Morphology_App/templates/human_smartphone_templates/File_4.png")
temp_sd1_5 = cv2.imread("/var/www/html/Sperm_Morphology_App/templates/human_smartphone_templates/File_5.png")
temp_sd1_6 = cv2.imread("/var/www/html/Sperm_Morphology_App/templates/human_smartphone_templates/File_6.png")
temp_sd1_7 = cv2.imread("/var/www/html/Sperm_Morphology_App/templates/human_smartphone_templates/File_7.png")
temp_sd1_8 = cv2.imread("/var/www/html/Sperm_Morphology_App/templates/human_smartphone_templates/File_8.png")
temp_sd1_9 = cv2.imread("/var/www/html/Sperm_Morphology_App/templates/human_smartphone_templates/File_9.png")
temp_sd1_10 = cv2.imread("/var/www/html/Sperm_Morphology_App/templates/human_smartphone_templates/File_10.png")
temp_sd1_11 = cv2.imread("/var/www/html/Sperm_Morphology_App/templates/human_smartphone_templates/File_11.png")
temp_sd1_12 = cv2.imread("/var/www/html/Sperm_Morphology_App/templates/human_smartphone_templates/File_12.png")


# template_list = [
#     ('temp_1_1', temp_1_1),
#     # ('temp_1_2',temp_1_2),
#     ('temp_1_3', temp_1_3),
#     ('temp_1_5', temp_1_5),
#     # ('temp_1_4',temp_1_4),
# ]

template_list = [
    ('temp_sd1_1',temp_sd1_1),
    ('temp_sd1_2',temp_sd1_2),
    ('temp_sd1_3',temp_sd1_3),
    ('temp_sd1_4',temp_sd1_4),
    ('temp_sd1_5',temp_sd1_5),
    ('temp_sd1_6',temp_sd1_6),
    ('temp_sd1_7',temp_sd1_7),
    ('temp_sd1_8',temp_sd1_8),
    ('temp_sd1_9',temp_sd1_9),
    ('temp_sd1_10',temp_sd1_10),
    ('temp_sd1_11',temp_sd1_11),
    ('temp_sd1_12',temp_sd1_12),

]
def extract_slides(root_dir, run_id,folder_name):
    input_path = root_dir + run_id + "/"+ folder_name + "/"
    output_path = root_dir + run_id + "/slides_heads/"

    slides = [slide for slide in os.listdir(input_path) if isdir(input_path + slide)]

    for slide in slides:

        print("Currently Extracting : ", slide)
        print()
        os.makedirs(output_path + slide, exist_ok=True)

        files = [input_path + slide + "/" + x for x in os.listdir(input_path + slide) if '.jpg' in x]
        # print("number_of_images", len(files))
        # print()

        number_of_images = len(files)
        print("Extracting from %d slices: " % number_of_images)

        for num in files:

            img = cv2.imread(num)

            try:
                listHit = matchTemplates(template_list, img, score_threshold=0.7, method=cv2.TM_CCOEFF_NORMED,
                                         maxOverlap=0.5)

                # continue

                print("Found {} hits".format(len(listHit)), "in image", os.path.basename(num))
                print()

                boxes = []
                for hit in listHit.index:
                    x, y, w, h = listHit['BBox'][hit]
                    # cv2.rectangle(img, (x, y), (x+w, y+h), [0,255,255], 1)
                    box = [x, y, x + w, y + h]
                    boxes.append(box)
                boxes = np.array(boxes)
                ctr = 1
                for b in boxes:
                    k = img[b[1]:b[3], b[0]:b[2]]
                    if k.shape[1] == h and k.shape[0] == w:
                        midpointX = (b[0] + b[2]) / 2
                        midpointY = (b[1] + b[3]) / 2
                        im_name = num.split("/")[-1][:-4]
                        # cv2.imwrite(path_1 + "%s_%d_%d.png" % (im_name, midpointX, midpointY), k)
                        # print(output_path + slide + '/' + slide + "__" + "%s_%d_%d.png" % (im_name, midpointX, midpointY))
                        cv2.imwrite(
                            output_path + slide + '/' + slide + "__" + "%s_%d_%d.png" % (
                                im_name, midpointX, midpointY), k)

                        ctr += 1
                    else:
                        pass
            except Exception as e:
                # print("error", e)
                print("")


if __name__ == '__main__':
    pass

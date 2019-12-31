import os
import cv2
import shutil
from data_helper_functions import filterImages
from xml.etree import ElementTree as ET
from albumentations import (
    HorizontalFlip, RandomBrightnessContrast, Compose, RandomRain, RandomSizedBBoxSafeCrop,
    RandomSunFlare, Cutout, IAAPerspective, ShiftScaleRotate, Flip, RandomGamma, IAAPiecewiseAffine, Rotate, RandomScale
)


def get_aug(aug, min_area=0., min_visibility=0.7):
    return Compose(aug, bbox_params={'format': 'pascal_voc', 'min_area': min_area, 'min_visibility': min_visibility,
                                     'label_fields': ['category_id']})


def do_augmentation(dataset_dir, output_dir, file_ext, strAugs):
    print("hi")

    for indDataset in dataset_dir:
        files_list = os.listdir(indDataset)
        imagesList = filterImages(files_list, file_ext)

        for augstr in strAugs:
            for image_name in imagesList:
                try:
                    base_name = os.path.splitext(image_name)[0]
                    image = cv2.imread(indDataset + image_name)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    tree = ET.parse(indDataset + base_name + ".xml")
                    root = tree.getroot()

                    bbox = []
                    label = []
                    idx = 0
                    xminxml = []
                    yminxml = []
                    xmaxxml = []
                    ymaxxml = []  # save to a new array to prevent out of order access

                    for xmin in root.iter('xmin'):
                        bbox.append([int(float(xmin.text))])
                        xminxml.append(xmin)

                    for ymin in root.iter('ymin'):
                        bbox[idx].append(int(float(ymin.text)))
                        idx += 1
                        yminxml.append(ymin)

                    idx = 0
                    for xmax in root.iter('xmax'):
                        bbox[idx].append(int(float(xmax.text)))
                        idx += 1
                        xmaxxml.append(xmax)

                    idx = 0
                    for ymax in root.iter('ymax'):
                        bbox[idx].append(int(float(ymax.text)))
                        idx += 1
                        ymaxxml.append(ymax)

                    idx = 0
                    for name in root.iter('name'):
                        label.append(name.text)
                        idx += 1

                    height, width, channels = image.shape
                    annotations = {'image': image.copy(), 'bboxes': bbox, 'category_id': label}
                    # category_id_to_name = {'car': 'car', 'tree': 'tree', 'house': 'house', 'pool':'pool'}
                    # category_id_to_name = {'car': 'car', 'truck': 'truck', 'bike': 'bike'}
                    category_id_to_name = {'cigarette': 'cigarette'}
                    # classNames = {1: 'parking sign',
                    #               2: 'stop sign',
                    #               3: 'tunnel sign',}
                    # visualize(annotations, category_id_to_name)
                    # plt.show()
                    ###augment the image
                    ###random crop
                    aug = get_aug([HorizontalFlip(p=0.5),
                                   # RandomSizedBBoxSafeCrop(height=300, width=300, p=0.3),
                                   RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                                   # RandomRain(blur_value=2, p=0.5),
                                   RandomSunFlare(p=0.3, src_radius=50),
                                   # Cutout(max_h_size=20, max_w_size=20, p=0.4),
                                   IAAPerspective(scale=(0.1, 0.1), p=0.4),
                                   # ShiftScaleRotate(scale_limit=0.2, border_mode=cv2.BORDER_CONSTANT, p=1.0),
                                   # RandomScale(p=0.3),
                                   # Rotate(p=0.3, border_mode=cv2.BORDER_CONSTANT, limit=30),
                                   # RandomGamma(p=1.0),
                                   IAAPiecewiseAffine(scale=(0.01, 0.01), p=0.4)
                                   ])
                    #
                    # aug = get_aug([HorizontalFlip(p=1)])
                    augmented = aug(**annotations)
                    # print(augmented)
                    # visualize(augmented, category_id_to_name)
                    # plt.show()
                    # print(augmented)
                    if augmented["bboxes"] == []: continue  # if it could not generate labels
                    print(len(augmented["bboxes"]))
                    idx = 0
                    for xmin in xminxml:
                        xmin.text = str(int(augmented['bboxes'][idx][0]))
                        idx += 1

                    idx = 0
                    for ymin in yminxml:
                        ymin.text = str(int(augmented['bboxes'][idx][1]))
                        idx += 1

                    idx = 0
                    for xmax in xmaxxml:
                        xmax.text = str(int(augmented['bboxes'][idx][2]))
                        idx += 1

                    idx = 0
                    for ymax in ymaxxml:
                        ymax.text = str(int(augmented['bboxes'][idx][3]))
                        idx += 1

                    for fileName in root.iter('filename'):
                        fileName.text = base_name + augstr + file_ext
                    # plt.show()

                    # write to file
                    tree.write(output_dir + base_name + augstr + ".xml")
                    image = cv2.cvtColor(augmented['image'], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(output_dir + base_name + augstr + file_ext, image)
                    print("saved image ", base_name + augstr + file_ext)

                except Exception as e:
                    print(e)
                    print("Exception: ", augmented)
                    # cv2.imshow("augmented", augmented["image"])
                    # cv2.waitKey(0)
    #
    ##Create validation file###
    # create_validation_file(output_dir)
    # print("Created Validation Files")
    #
    print("copying original")
    ###Copy original into source directory
    for folder in dataset_dir:
        filenames = os.listdir(folder)
        print(filenames)
        for name in filenames:
            print("in: ", folder + name)
            print("out: ", output_dir + name)
            print("name:", name)
            shutil.copy(folder + name, output_dir + name)
            print("moved: " + name)
    print("Done")


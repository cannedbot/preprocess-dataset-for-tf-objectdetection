import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    files = os.listdir(path)
    # print(files)
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    # image_train_path = os.path.join("D:/model_master_original/models-master/research/object_detection/datasets/Traffic_Light/CombinedAndAugmented")
    # image_test_path = os.path.join("D:/model_master_original/models-master/research/object_detection/datasets/Traffic_Light/CombinedAndAugmented/Validation")
    #
    # xml_df = xml_to_csv(image_train_path)
    # xml_df.to_csv('traffic_light_train_aug.csv', index=None)
    #
    # xml_df = xml_to_csv(image_test_path)
    # xml_df.to_csv('traffic_light_test_aug.csv', index=None)
    #
    # print('Successfully converted xml to csv for both train and test.')

    image_train_path = os.path.join(
        "D:/model_master_original/models-master/research/object_detection/images_macncheese/train")
    image_test_path = os.path.join(
        "D:/model_master_original/models-master/research/object_detection/images_macncheese/test")

    xml_df = xml_to_csv(image_train_path)
    xml_df.to_csv('D:/model_master_original/models-master/research/object_detection/images_macncheese/cheese_train.csv', index=None)

    xml_df = xml_to_csv(image_test_path)
    xml_df.to_csv('D:/model_master_original/models-master/research/object_detection/images_macncheese/cheese_test.csv', index=None)

    print('Successfully converted xml to csv for both train and test.')

#
# main()
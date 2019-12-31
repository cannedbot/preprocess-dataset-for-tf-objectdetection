from generate_augmentation import *
from export_to_csv import xml_to_csv
import tensorflow as tf
import pandas as pd
from generate_tfrecord import class_text_to_int, split, create_tf_example


DoAugmentation = True
generate_test_only = False

BOX_COLOR = (255, 0, 0)
TEXT_COLOR = (255, 255, 255)

dataset_dir = [
    "E:/Repos/preprocess-dataset-for-tf-objectdetection/dataset/"
]
output_dir = "E:/Repos/preprocess-dataset-for-tf-objectdetection/aug_dataset/"  # don't forget slash

image_test_path = "E:/Repos/preprocess-dataset-for-tf-objectdetection/dataset-aug-test/"
image_train_path = "E:/Repos/preprocess-dataset-for-tf-objectdetection/aug_dataset/"

tf_record_train = "train.record"
tf_record_test = "test.record"

csv_test = 'test.csv'
csv_train = 'train.csv'

file_ext = ".jpg"
string_aug_name = ["aug1"]  # string to append on augmented images

###Main
# Functions to visualize bounding boxes and class labels on an image.
# Based on https://github.com/facebookresearch/Detectron/blob/master/detectron/utils/vis.py

if __name__ == '__main__':

    if DoAugmentation:
        do_augmentation(dataset_dir, output_dir, file_ext, strAugs=string_aug_name)

    if not generate_test_only:

        xml_df = xml_to_csv(image_train_path)  # for train
        xml_df.to_csv(csv_train, index=None)

        print('Successfully converted xml to csv for both train and test.')

        writer = tf.python_io.TFRecordWriter(tf_record_train)
        path = os.path.join(image_train_path)
        examples = pd.read_csv(csv_train)
        grouped = split(examples, 'filename')
        for group in grouped:
            print("generating: ", group)
            tf_example = create_tf_example(group, path)
            writer.write(tf_example.SerializeToString())
        ###



        writer.close()
        output_path = os.path.join(os.getcwd(), tf_record_train)
        print('Successfully created the TFRecords: {}'.format(output_path))



    xml_df = xml_to_csv(image_test_path)  # for test
    xml_df.to_csv(csv_test, index=None)

    writer = tf.python_io.TFRecordWriter(tf_record_test)
    path = os.path.join(image_test_path)
    examples = pd.read_csv(csv_test)
    grouped = split(examples, 'filename')
    for group in grouped:
        print("generating: ", group)
        tf_example = create_tf_example(group,
                                       path)  # be careful with this ### make sure you change the class_text_to_int to have proper labels first
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), tf_record_test)
    print('Successfully created the TFRecords: {}'.format(output_path))

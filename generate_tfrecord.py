"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=train.record --image_dir=path/to/imagepath
  # Create test data:
  python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=test.record
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
import tensorflow as tf

from PIL import Image
import dataset_util
from collections import namedtuple, OrderedDict

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('image_dir', '', 'Path to images')
FLAGS = flags.FLAGS


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'cigarette':
        return 1
    # elif row_label == 'truck':
    #     return 2
    # elif row_label == 'bike':
    #     return 3
    # # elif row_label == 'pool':
    # #     return 4
    else:
        assert False, ("Error, row label incorrect, not a part of any of the specified labels: " + row_label)
    # if row_label == 'macncheese':
    #     return 1
    # else:
    #     assert False, ("Error, row label incorrect, not a part of any of the specified labels: " + row_label)

# def class_text_to_int(row_label):
#     if row_label == 'green':
#         return 1
#     elif row_label == 'red':
#         return 2
#     elif row_label == 'yellow_light':
#         return 3
#     elif row_label == 'parking sign':
#         return 4
#     elif row_label == 'stop sign':
#         return 5
#     elif row_label == 'tunnel sign':
#         return 6
#     elif row_label == 'barrier':
#         return 7
#     else:
#         assert False, ("Error, row label incorrect, not a part of any of the specified labels: " + row_label)

# def class_text_to_int(row_label):
#     row_label = str(row_label)
#     if row_label == '1':
#         return 1
#     elif row_label == '2':
#         return 2
#     elif row_label == '3':
#         return 3
#     elif row_label == '4':
#         return 4
#     elif row_label == '5':
#         return 5
#     elif row_label == '6':
#         return 6
#     elif row_label == '7':
#         return 7
#     elif row_label == '8':
#         return 8
#     elif row_label == '9':
#         return 9
#     elif row_label == '10':
#         return 10
#     elif row_label == '11':
#         return 11
#     elif row_label == '12':
#         return 12
#     elif row_label == '13':
#         return 13
#     elif row_label == '14':
#         return 14
#     elif row_label == '15':
#         return 15
#     elif row_label == '16':
#         return 16
#     elif row_label == '17':
#         return 17
#     elif row_label == '18':
#         return 18
#     elif row_label == '19':
#         return 19
#     elif row_label == '20':
#         return 20
#     elif row_label == '21':
#         return 21
#     elif row_label == '22':
#         return 22
#     elif row_label == '23':
#         return 23
#     elif row_label == '24':
#         return 24
#     elif row_label == '25':
#         return 25
#     elif row_label == '26':
#         return 26
#     elif row_label == '27':
#         return 27
#     elif row_label == '28':
#         return 28
#     elif row_label == '29':
#         return 29
#     elif row_label == '30':
#         return 30
#     elif row_label == '31':
#         return 31
#     elif row_label == '32':
#         return 32
#     elif row_label == '33':
#         return 33
#     elif row_label == '34':
#         return 34
#     elif row_label == '35':
#         return 35
#     elif row_label == '36':
#         return 36
#     elif row_label == '37':
#         return 37
#     elif row_label == '38':
#         return 38
#     elif row_label == '39':
#         return 39
#     elif row_label == '40':
#         return 40
#     elif row_label == '41':
#         return 41
#     elif row_label == '42':
#         return 42
#     elif row_label == '43':
#         return 43
#     elif row_label == '44':
#         return 44
#     elif row_label == '45':
#         return 45
#     elif row_label == '46':
#         return 46
#     elif row_label == '47':
#         return 47
#     elif row_label == '48':
#         return 48
#     elif row_label == '49':
#         return 49
#     elif row_label == '50':
#         return 50
#     elif row_label == '51':
#         return 51
#     elif row_label == '52':
#         return 52
#     elif row_label == '53':
#         return 53
#     elif row_label == '54':
#         return 54
#     elif row_label == '55':
#         return 55
#     elif row_label == '56':
#         return 56
#     elif row_label == '57':
#         return 57
#     elif row_label == '58':
#         return 58
#     elif row_label == '59':
#         return 59
#     elif row_label == '60':
#         return 60
#     elif row_label == '61':
#         return 61
#     elif row_label == '62':
#         return 62
#     elif row_label == '63':
#         return 63
#     elif row_label == '64':
#         return 64
#     elif row_label == '65':
#         return 65
#     elif row_label == '66':
#         return 66
#     elif row_label == '67':
#         return 67
#     elif row_label == '68':
#         return 68
#     elif row_label == '69':
#         return 69
#     elif row_label == '70':
#         return 70
#     elif row_label == '71':
#         return 71
#     elif row_label == '72':
#         return 72
#     elif row_label == '73':
#         return 73
#     elif row_label == '74':
#         return 74
#     elif row_label == '75':
#         return 75
#     elif row_label == '76':
#         return 76
#     elif row_label == '77':
#         return 77
#     elif row_label == '78':
#         return 78
#     elif row_label == '79':
#         return 79
#     elif row_label == '80':
#         return 80
#     elif row_label == '81':
#         return 81
#     elif row_label == '82':
#         return 82
#     elif row_label == '83':
#         return 83
#     elif row_label == '84':
#         return 84
#     elif row_label == '85':
#         return 85
#     elif row_label == '86':
#         return 86
#     elif row_label == '87':
#         return 87
#     elif row_label == '88':
#         return 88
#     elif row_label == '89':
#         return 89
#     elif row_label == '90':
#         return 90
#     elif row_label == '91':
#         return 91
#     elif row_label == '92':
#         return 92
#     elif row_label == '93':
#         return 93
#     elif row_label == '94':
#         return 94
#     elif row_label == '95':
#         return 95
#     elif row_label == '96':
#         return 96
#     elif row_label == '97':
#         return 97
#     elif row_label == '98':
#         return 98
#     elif row_label == '99':
#         return 99
#     elif row_label == '100':
#         return 100
#     else:
#         assert False, ("Error, row label incorrect, not a part of any of the specified labels: " + row_label)


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(str(row['class']).encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    path = os.path.join(FLAGS.image_dir)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        print("generating: ", group)
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())

    writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


# if __name__ == '__main__':
#     tf.app.run()
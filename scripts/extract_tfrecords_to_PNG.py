import tensorflow.compat.v1 as tf
import numpy as np
import PIL.Image
import PIL.ImageFile

def parse_tfrecord_np(record):
    ex = tf.train.Example()
    ex.ParseFromString(record)
    shape = ex.features.feature['shape'].int64_list.value # temporary pylint workaround # pylint: disable=no-member
    data = ex.features.feature['data'].bytes_list.value[0] # temporary pylint workaround # pylint: disable=no-member
    return np.fromstring(data, np.uint8).reshape(shape)

if __name__ == "__main__":
    tfr_file = "../ffhq-r08.tfrecords"
    output_folder = 'output_folder_path'
    verbose = True
    buffer_mb = 256

    tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
    for i, record in enumerate(tf.python_io.tf_record_iterator(tfr_file, tfr_opt)):
        np_img = parse_tfrecord_np(record).transpose(1,2,0)
        img = PIL.Image.fromarray(np_img, 'RGB')

        img_name = str(i).zfill(5)

        img.save(output_folder + img_name + '.png')

        if verbose and i % 100 == 0:
            print('step:', i)

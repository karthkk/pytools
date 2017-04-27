import tensorflow as tf
import numpy as np
import cv2

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

class TfExample():
   
    def __init__(self, example = None):
        if example:
            self.feature = example.features.feature
        else:
            self.feature = {}

    def update_example(self, example):
        self.feature = example.features.feature

    def get_example(self):
        return tf.train.Example(features=tf.train.Features(feature=self.feature))
    
    def add_img(self, key, img):
        self.unwritten[key] = _bytes_feature(img.tostring())
        img_sz = img.shape
        self.add_int_arr(key+'_sz',img_sz)

    def add_int_arr(self, key, val):
        self.feature[key] = _int64_feature(val)

    def add_float_arr(self, key, val):
        self.feature[key] = _float_feature(val)

    def add_str(self, key, val):
        self.feature[key] = _bytes_feature(val)

    def read_img(self, key):
        feature = self.feature
        image_str = feature[name].bytes_list.value[0]
        image_sz = feature[name].int64_list.value
        arr = np.fromstring(image_str,dtype=np.uint8).reshape((image_sz[0], image_sz[1], image_sz[2]))
        return arr

    def read_int_arr(self, key):
        return feature[name].int64_list.value
       
    def read_float_arr(self, key):
        return feature[name].float_list.value

class TFDataReader():

    def __init__(self, path):
        self.reader =  tf.python_io.tf_record_iterator(path)
        self.example = tf.train.Example()

   
class TFDataWriter():

    def __init__(self, path):
        self.writer = tf.python_io.TFRecordWriter(path)
        self.example = tf.train.Example()

    def  

    def next(self):
        
    
    def save(self):
        pass


    def read_img



import tensorflow as tf
import numpy as np



IMG_TYP = 'IMG'
INT_TYP = "INT"
FLT_TYP = "FLT"
STR_TYP = "STR"

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def read_img(feature, key):
    image_str = feature[key].bytes_list.value[0]
    image_sz = feature[key + '_sz'].int64_list.value
    arr = np.fromstring(image_str,dtype=np.uint8).reshape((image_sz[0], image_sz[1], image_sz[2]))
    return arr

def read_int_arr(feature, key):
    return feature[key].int64_list.value

def read_float_arr(feature, key):
    return feature[key].float_list.value

def read_str(feature, key):
    return feature[key].bytes_list.value[0]

def add_int_arr(dct, ky, val):
    dct[ky] = _int64_feature(val)

def add_float_arr(dct, ky, val):
    dct[ky] = _float_feature(val)

def add_str(dct, ky, val):
    dct[ky] = _bytes_feature(val)

def add_img(dct, ky, val):
    dct[ky] = _bytes_feature(val.tostring())
    img_sz = val.shape
    add_int_arr(dct, ky + '_sz', img_sz)

reader_lookup_by_type = {INT_TYP: read_int_arr, FLT_TYP: read_float_arr, IMG_TYP: read_img, STR_TYP: read_str}
writer_lookup_by_type = {INT_TYP: add_int_arr, FLT_TYP: add_float_arr, IMG_TYP: add_img, STR_TYP: add_str}

class TFDataReader():

    def __init__(self, path):
        self.reader =  tf.python_io.tf_record_iterator(path)
        self.example = tf.train.Example()

    def read(self, ky_typ):
        self.example.ParseFromString(self.reader.next())
        dct = {}
        feature = self.example.features.feature
        for (ky,typ) in ky_typ:
            lookup_fn = reader_lookup_by_type.get(typ, None)
            if lookup_fn is None:
                raise ValueError('%s is not a valid type' % typ)
            dct[ky] = lookup_fn(feature, ky)
        return dct
   
class TFDataWriter():

    def __init__(self, path):
        self.writer = tf.python_io.TFRecordWriter(path)

    def add(self, ky_val_typ):
        dct = {}
        for (ky, val, typ) in ky_val_typ:
            add_fn = writer_lookup_by_type.get(typ, None)
            if add_fn is None:
                raise ValueError('%s is not a valid type'%typ)
            add_fn(dct, ky, val)

        example = tf.train.Example(features=tf.train.Features(feature=dct))
        self.writer.write(example.SerializeToString())

    def close(self):
        self.writer.close()
    
if __name__ == "__main__":
    img = np.zeros((20,20,3), dtype=np.uint8)
    inta = [1,2,3]
    flta = [2., 1., 5.]
    tstr = 'test a'
    writer = TFDataWriter('/tmp/test')
    writer.add([ ( 'a', img, IMG_TYP), ('b', flta, FLT_TYP), ('c', inta, INT_TYP), ('d', tstr, STR_TYP)])
    writer.close()
    reader = TFDataReader('/tmp/test')
    dat = reader.read([('a', IMG_TYP), ('b', FLT_TYP), ('c', INT_TYP), ('d', STR_TYP)])
    assert(dat['a'].shape == (20,20,3))
    (v1, v2, v3) = dat['b']
    assert (v1 == 2.)
    assert (v2 == 1.)
    assert (v3 == 5.)
    (v1, v2, v3) = dat['c']
    assert (v1 == 1)
    assert (v2 == 2)
    assert (v3 == 3)
    assert(dat['d'] == 'test a')










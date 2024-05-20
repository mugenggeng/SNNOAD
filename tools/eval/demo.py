'''
Convert a pkl file into json file
'''
import sys
import os
import pickle
import json
import numpy



class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
                            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
                            numpy.uint16, numpy.uint32, numpy.uint64)):
            return int(obj)
        elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32,
                              numpy.float64)):
            return float(obj)
        elif isinstance(obj, (numpy.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def convert_dict_to_json(file_path):
    with open(file_path, 'rb') as fpkl, open('%s.json' % file_path, 'w') as fjson:
        data = pickle.load(fpkl,encoding='latin1')
        json.dump(data, fjson, ensure_ascii=False, sort_keys=True, indent=4,cls=NumpyEncoder)




def main():
    # if sys.argv[1] and os.path.isfile(sys.argv[1]):
        file_path = './ELGC.pkl'
        print("Processing %s ..." % file_path)
        convert_dict_to_json(file_path)
    # else:
    #     print("Usage: %s abs_file_path" % (__file__))


if __name__ == '__main__':
    main()

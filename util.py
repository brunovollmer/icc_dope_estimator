import cv2
import numpy as np
import json

def format_debug_times(timestamps):
    print("-----------------------")
    for i in range(len(timestamps)-1):
        start_time = timestamps[i]['time']
        end_time = timestamps[i+1]['time']
        duration = end_time - start_time
        print("{} --> {}: {} seconds".format(timestamps[i]['event'], timestamps[i+1]['event'], duration))
    print("-----------------------")

def resize_image(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)


def save_json(path, data):
    with open(path, 'w') as outfile:
        json.dump(data, outfile, cls=NumpyEncoder)

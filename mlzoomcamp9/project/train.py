import numpy as np

from io import BytesIO
from urllib import request
from PIL import Image

import tflite_runtime.interpreter as tflite


# load the tflite model
interpreter = tflite.Interpreter(model_path="cats-dogs-v2.tflite")
# load the weights too
interpreter.allocate_tensors()


# get input index and output index
input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]['index']


# download, resize  and preprocess input function
def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


def preprocess_input(x):
    x /= 255.0
    return x


# url = "https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg"


def predict_image(url):
    image = download_image(url)
    img = prepare_image(image, target_size=(150, 150))

    # preprocess
    x = np.array(img, dtype="float32")
    X = np.array([x])
    X = preprocess_input(X)

    # model
    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    # make prediction
    preds = interpreter.get_tensor(output_index)

    # change prediction to list

    preds = preds[0].tolist()
    return preds


def lambda_handler(event, context):
    url = event['url']
    result = predict_image(url)
    return result

import os, sys, json, pickle, base64, time
import numpy as np
import PIL
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from io import BytesIO

#-------------------------------------------------------------------------------
# Set up server.
app = Flask(__name__, static_url_path='')
CORS(app)

#-------------------------------------------------------------------------------
# Init TF.
tflib.init_tf()
sess = tf.get_default_session()
synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

#-------------------------------------------------------------------------------
# Config.
with open('./config.json', 'r') as json_file:
    config = json.load(json_file)

assert len(config['models']), "No models given in config."

models = {}
for model in config['models']:
    _, _, Gs = pickle.load(open(model['path'], 'rb'))
    models[model['name']] = Gs
    print("Loaded", model['name'])

#-------------------------------------------------------------------------------
# Utils.
def encode_jpeg_bytes(arr, quality=90):
    image = PIL.Image.fromarray(arr, 'RGB')
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    return buffered.getvalue()

def encode_jpeg_string(arr):
    img_bytes = base64.b64encode(encode_jpeg_bytes(arr))
    return img_bytes.decode('ascii')

def ondone(request, images, zlatents):
    jpegs = list(map(encode_jpeg_string, images))
    return json.dumps([ jpegs, zlatents.tolist() ])

#-------------------------------------------------------------------------------
# Image Utils.

def make_images(model, zlatents):
    Gs = models[model]
    return Gs.run(zlatents, None, truncation_psi=0.7, randomize_noise=False, **synthesis_kwargs)

def random_zlatents(model, num):
    return np.random.randn(num, models[model].input_shape[1])


#-------------------------------------------------------------------------------
# Routes.
@app.route('/hello', methods=['POST'])
def post_hello():
    return "Hello World"

@app.route('/random', methods=['POST'])
def post_random():
    try:
        if (not tf.get_default_session()):
            sess.__enter__()
        time_start = time.time()
        num      = int(request.form.get('num', 8))
        model    = request.form.get('model')
        zlatents = random_zlatents(model, num)
        images   = make_images(model, zlatents)
        result   = ondone(request, images, zlatents)
        print('Made %i random in %f seconds.' % (num, time.time() - time_start))
        return result
    except Exception as e:
        print('Error in /random', e)
        return '', 500

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    app.run(host='0.0.0.0', debug=True, port=port, threaded=False)

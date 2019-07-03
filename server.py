import json, pickle, time
import numpy as np
import PIL
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import tensorflow as tf
from io import BytesIO
import ai_integration

# -------------------------------------------------------------------------------
# Init TF.
tflib.init_tf()
sess = tf.get_default_session()
synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

# -------------------------------------------------------------------------------
# Config.
with open('./config.json', 'r') as json_file:
    config = json.load(json_file)

assert len(config['models']), "No models given in config."

models = {}
for model in config['models']:
    _, _, Gs = pickle.load(open(model['path'], 'rb'))
    models[model['name']] = Gs
    print("Loaded", model['name'])


# -------------------------------------------------------------------------------
# Utils.
def encode_jpeg_bytes(arr, quality=95):
    image = PIL.Image.fromarray(arr, 'RGB')
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=quality)
    return buffered.getvalue()


# -------------------------------------------------------------------------------
# Image Utils.

def make_images(model, zlatents):
    Gs = models[model]
    return Gs.run(zlatents, None, truncation_psi=0.7, randomize_noise=False, **synthesis_kwargs)


def random_zlatents(model, num):
    return np.random.randn(num, models[model].input_shape[1])


while True:
    with ai_integration.get_next_input(inputs_schema={
        # no inputs yet
        'latent_vector': {
            'type': 'text'
        }
    }) as inputs_dict:

        # only update the negative fields if we reach the end of the function - then update successfully
        result_data = {"content-type": 'text/plain',
                       "data": None,
                       "success": False,
                       "error": None}

        if (not tf.get_default_session()):
            sess.__enter__()
        time_start = time.time()
        num = 1
        model = 'celebhq'

        if inputs_dict['latent_vector'] == 'random':

            # generate new random vector
            zlatents = random_zlatents(model, num)
        else:
            # use passed vector
            vector = json.loads(inputs_dict['latent_vector'])

            if len(vector) != 512:
                raise Exception('Input vector must be length 512, floating point numbers.')

            zlatents = [vector]

        images = make_images(model, zlatents)
        print('Made random in %f seconds.' % (time.time() - time_start))
        result_data["content-type"] = 'image/jpeg'
        result_data["data"] = encode_jpeg_bytes(images[0])
        result_data["success"] = True
        result_data["error"] = None

        ai_integration.send_result(result_data)

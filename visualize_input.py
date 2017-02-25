import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import videoto3d
from sklearn.cross_validation import train_test_split
from tqdm import tqdm

from keras.datasets import cifar10
from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils.visualize_util import plot
from keras import backend as K


class LearningRateAdaptor():

    def __init__(self, grad_shape, decay_rate=0.95, beta1=0.9, beta2=0.999):
        self.cache = np.zeros(grad_shape)
        self.cache1 = np.zeros(grad_shape)
        self.cache2 = np.zeros(grad_shape)
        self.decay_rate = decay_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.iteration = 0

    def rmsprop(self, grads):
        self.cache = self.decay_rate * self.cache + \
            (1 - self.decay_rate) * self.grads ** 2
        step = grads / np.sqrt(self.cache + K.epsilon())

        return step

    def adam(self, grads):

        self.cache1 = self.beta1 * self.cache1 + (1 - self.beta1) * grads
        self.cache2 = self.beta2 * self.cache2 + (1 - self.beta2) * grads**2
        self.cache1_h = self.cache1 / (1 - self.beta1**(self.iteration + 1))
        self.cache2_h = self.cache2 / (1 - self.beta2**(self.iteration + 1))
        step = self.cache1 / np.sqrt(self.cache2 + K.epsilon())
        self.iteration += 1

        return step


def generateInputImage(model, layer_name, filter_index, niterations=20):
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    input_shape = model.input_shape
    input_img = model.layers[0].input

    layer_output = layer_dict[layer_name].output
    if len(layer_dict[layer_name].output_shape) == 2:
        # layer_output(dense) : (nsamples, m th output)
        activation = K.mean(layer_output[:, filter_index])
    elif len(layer_dict[layer_name].output_shape) == 5:
        # layer_output(3d) : (nsamples, dim1, dim2, dim3, channels)
        activation = K.mean(layer_output[:, :, :, :, filter_index])
    else:
        # layer_output(2d) : (nsamples, rows, cols, channels)
        activation = K.mean(layer_output[:, :, :, filter_index])

    grads = K.gradients(activation, input_img)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + K.epsilon())

    iterate = K.function([input_img, K.learning_phase()], [activation, grads])

    x = np.random.random((1, ) + input_shape[1:])
    x = (x - 0.5) * 20 + 128

    adaptor = LearningRateAdaptor(K.int_shape(grads)[1:])

    for i in tqdm(range(niterations)):
        activation_value, grads_value = iterate([x, 0])
        step = adaptor.adam(grads_value)
        x += step

    test_maximized(model, x)
    img = x[0]
    if len(input_shape) == 5:
        for i in range(img.shape[2]):
            frame = deprocess_image(img[:, :, i, :])
            save_figure(frame, '{}_{}_{}'.format(layer_name, filter_index, i))
    else:
        img = deprocess_image(img)
        save_figure(img, '{}_{}'.format(layer_name, filter_index))


def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')

    return x


def test_maximized(model, img):
    prediction = model.predict(img, 1)
    print('class {}: {}'.format(np.argmax(prediction), np.max(prediction)))


def save_figure(img, name):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img[:, :, 0], cmap='gray')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    fig.savefig('maximize_{}.png'.format(name))


def main():
    parser = argparse.ArgumentParser(
        description='Implementaiton of visualizing input image')
    parser.add_argument('--model', '-m', type=str,
                        required=True, help='json of model shape')
    parser.add_argument('--weights', '-w', type=str,
                        required=True, help='hd5 of weights')
    parser.add_argument('--layernames', '-l', type=bool,
                        default=False, help='show layer names')
    parser.add_argument('--name', '-n', type=str, required=True,
                        help='the name of layer which will be maximized.')
    parser.add_argument('--index', '-i', type=int, required=True,
                        help='index of layer output which will be maximized.')
    parser.add_argument('--iter', type=int, default=20,
                        help='the number of iterations.')
    args = parser.parse_args()

    with open(args.model, 'r') as model_json:
        model = model_from_json(model_json.read())
    model.load_weights(args.weights)

    if args.layernames:
        model.summary()
        return 0

    generateInputImage(model, args.name, args.index, args.iter)

if __name__ == '__main__':
    main()

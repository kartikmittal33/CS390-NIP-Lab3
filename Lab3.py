import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import random
from scipy.misc import imsave, imresize
from scipy.optimize import \
    fmin_l_bfgs_b  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import warnings

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

random.seed(1618)
np.random.seed(1618)
# tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

CONTENT_IMG_PATH = "gate"
STYLE_IMG_PATH = "composition"

CONTENT_IMG_H = 500
CONTENT_IMG_W = 500

STYLE_IMG_H = 500
STYLE_IMG_W = 500

CONTENT_WEIGHT = 0.005  # Alpha weight.
STYLE_WEIGHT = 7.0  # Beta weight.
TOTAL_WEIGHT = 1.0
LOSS_WEIGHT = 1.25

TRANSFER_ROUNDS = 5

NUM_OF_FILTERS = 3

# =============================<Helper Fuctions>=================================
'''
TODO: implement this.
This function should take the tensor and re-convert it to an image.
'''


def deprocessImage(img):
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    x = img[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def gramMatrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


# ========================<Loss Function Builder Functions>======================

def styleLoss(style, gen):
    return K.sum(K.square(gramMatrix(style) - gramMatrix(gen))) / (4. * (NUM_OF_FILTERS ** 2) * (CONTENT_IMG_H * CONTENT_IMG_W) ** 2)


def contentLoss(content, gen):
    return K.sum(K.square(gen - content))


def totalLoss(x):
    a = K.square(
        x[:, :CONTENT_IMG_H - 1, :CONTENT_IMG_W - 1, :] -
        x[:, 1:, :CONTENT_IMG_W - 1, :])
    b = K.square(
        x[:, :CONTENT_IMG_H - 1, :CONTENT_IMG_W - 1, :] -
        x[:, :CONTENT_IMG_H - 1, 1:, :])
    return K.sum(K.pow(a + b, LOSS_WEIGHT))


# =========================<Pipeline Functions>==================================

def getRawData():
    content_path = CONTENT_IMG_PATH + ".jpg"
    style_path = STYLE_IMG_PATH + ".jpg"
    print("   Loading images.")
    print("      Content image URL:  \"%s\"." % content_path)
    print("      Style image URL:    \"%s\"." % style_path)
    cImg = load_img(content_path)
    tImg = cImg.copy()
    sImg = load_img(style_path)
    print("      Images have been loaded.")
    return (
    (cImg, CONTENT_IMG_H, CONTENT_IMG_W), (sImg, STYLE_IMG_H, STYLE_IMG_W), (tImg, CONTENT_IMG_H, CONTENT_IMG_W))


def preprocessData(raw):
    img, ih, iw = raw
    img = img_to_array(img)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = imresize(img, (ih, iw, 3))
    img = img.astype("float64")
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


class Evaluator(object):

    def __init__(self, fetch_loss_and_grads):
        self.loss_value = None
        self.grads_values = None
        self.fetch_loss_and_grads = fetch_loss_and_grads

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
        outs = self.fetch_loss_and_grads([x])

        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


'''
TODO: Allot of stuff needs to be implemented in this function.
First, make sure the model is set up properly.
Then construct the loss function (from content and style loss).
Gradient functions will also need to be created, or you can use K.Gradients().
Finally, do the style transfer with gradient descent.
Save the newly generated and deprocessed images.
'''


def styleTransfer(cData, sData, tData):
    print("   Building transfer model.")
    contentTensor = K.variable(cData)
    styleTensor = K.variable(sData)
    genTensor = K.placeholder((1, CONTENT_IMG_H, CONTENT_IMG_W, 3))
    inputTensor = K.concatenate([contentTensor, styleTensor, genTensor], axis=0)
    model = vgg19.VGG19(include_top=False, weights="imagenet", input_tensor=inputTensor)
    outputDict = dict([(layer.name, layer.output) for layer in model.layers])
    print("   VGG19 model loaded.")
    loss = 0.0
    styleLayerNames = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
    contentLayerName = "block5_conv2"
    print("   Calculating content loss.")
    contentLayer = outputDict[contentLayerName]
    contentOutput = contentLayer[0, :, :, :]
    genOutput = contentLayer[2, :, :, :]
    loss += contentLoss(contentOutput, genOutput) * CONTENT_WEIGHT  # TODO: implement.
    print("   Calculating style loss.")
    for layerName in styleLayerNames:
        styleLayer = outputDict[layerName]
        styleOutput = styleLayer[1, :, :, :]
        genOutput = styleLayer[2, :, :, :]
        loss += styleLoss(styleOutput, genOutput) * STYLE_WEIGHT  # TODO: implement.
    loss += totalLoss(genTensor) * TOTAL_WEIGHT  # TODO: implement.
    # TODO: Setup gradients or use K.gradients().
    gradients = K.gradients(loss, genTensor)[0]
    fetch_loss_and_grads = K.function([genTensor], [loss, gradients])

    evaluator = Evaluator(fetch_loss_and_grads)
    x = tData.flatten()

    print("   Beginning transfer.")
    for i in range(TRANSFER_ROUNDS):
        print("   Step %d." % i)
        # TODO: perform gradient descent using fmin_l_bfgs_b.

        x, tLoss, info = fmin_l_bfgs_b(evaluator.loss,
                                         x,
                                         fprime=evaluator.grads,
                                         maxfun=500)

        print("      Loss: %f." % tLoss)
        img = x.copy().reshape((CONTENT_IMG_H, CONTENT_IMG_W, 3))
        img = deprocessImage(img)
        saveFile = "%s_%s_%d.jpg" % (CONTENT_IMG_PATH, STYLE_IMG_PATH, i)
        imsave(saveFile, img)   #Uncomment when everything is working right.
        print("      Image saved to \"%s\"." % saveFile)
    print("   Transfer complete.")


# =========================<Main>================================================

def main():
    print("Starting style transfer program.")
    raw = getRawData()
    cData = preprocessData(raw[0])  # Content image.
    sData = preprocessData(raw[1])  # Style image.
    tData = preprocessData(raw[2])  # Transfer image.
    styleTransfer(cData, sData, tData)
    print("Done. Goodbye.")


if __name__ == "__main__":
    main()

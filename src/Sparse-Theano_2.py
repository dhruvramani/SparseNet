import numpy as np
import theano.tensor as T
from theano import shared, function
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Formula Y' = 0.299 R + 0.587 G + 0.114 B
# Courtesy : http://stackoverflow.com/a/12201744/4534903
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def getData(batchSize):
    data = np.zeros((batchSize, 1024))
    print("=> Generating Data.")
    for i in range(1,batchSize):
        img = rgb2gray(mpimg.imread('../data/airplane/{num:04d}.png'.format(num=i)))
        data[i, : ] = np.ndarray.flatten(img)
    print("=> Data Ready.")
    return data

def train(data, noComp, layerNo):
    alpha = 0.01
    X = T.dmatrix()
    lambdaC = T.dscalar()
    D = shared(np.zeros((noComp, X.shape[1])))
    W = shared(np.zeros(X.shape[0], noComp))
    
    gen = np.dot(W, D)
    genFunc = function([], gen)
    cost = np.sum(np.sqaure(X - np.dot(W, D))) + lambdaC * np.sum(np.square(W))
    costFunc = function([X, lambdaC], cost, updates = [(W, W - alpha * T.grad(cost, W)), (D, D - alpha * T.grad(cost, D))])

    for i in range(1000):
        cost = costFunc(data, 0.1)
        if i%50 == 0:
            print("== Cost : {}".format(cost))
    
    pickle.dump(D, open('../weights/dictionary_theano_{}.sav'.format(layerNo), 'wb'))
    print("=> Dictionary Learnt and Saved.")
    pickle.dump(W, open('../weights/weights_theano_{}.sav'.format(layerNo), 'wb'))
    print("=> Weights Learnt and Saved.")

def test(testImg, noComp, layerNo):
    alpha = 0.01
    X = T.dmatrix()
    lambdaC = T.dscalar()
    D = pickle.load(open('../weights/dictionary_theano_{}.sav'.format(layerNo), 'rb'))
    W = shared(np.zeros(X.shape[0], noComp))

    gen = np.dot(W, D)
    genFunc = function([], gen)
    cost = np.sum(np.sqaure(X - np.dot(W, D))) + lambdaC * np.sum(np.square(W))
    costFunc = function([X, lambdaC], cost, updates = [(W, W - alpha * T.grad(cost, W))])

    for i in range(1000):
        cost = costFunc(testImg, 0.1)
    image = np.reshape(gen(), (32, 32))
    plt.imshow(image)
    plt.savefig('../output/test_theano_{}.png'.format(layerNo))
    print("=> Weights learnt and Image Saved for Test-Layer {}.".format(layerNo))

if __name__ == '__main__':
    images = getData(432)
    images = train(images, 900, 1)
    images = train(images, 900, 2)
    images = train(images, 900, 3)

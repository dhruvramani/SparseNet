import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import sparse_encode, MiniBatchDictionaryLearning

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

# TODO : Argument Parser to decide which Weights to use
# Weights/Dictionary of Cars : dictionary_cars_1.sav, Airplanes + Cars : dictionary_airplaneCars_1.sav
def test(imagePath, noLayers):
    img = rgb2gray(mpimg.imread(imagePath))
    X = np.ndarray.flatten(img)
    for i in range(1, noLayers+1):
        mbdl = pickle.load(open('../weights/dictionary_airplane_{}.sav'.format(i), 'rb'))
        print("=> Learning Weights for Test-Layer {}.".format(i))
        code = sparse_encode(X, mbdl.components_)
        newX = np.dot(code, mbdl.components_)
        image = np.reshape(newX, (32, 32))
        X = newX
        plt.imshow(image)
        plt.savefig('../output/test_airplane_{}.png'.format(i))
        print("=> Weights learnt and Image Saved for Test-Layer {}.".format(i))

def train(X, noComp, imageToDisplay, layerNo):
    print("=> Learning Dictionary for Layer {}.".format(layerNo))
    mbdl = MiniBatchDictionaryLearning(noComp)
    mbdl.fit(X)
    pickle.dump(mbdl, open('../weights/dictionary_airplane_{}.sav'.format(layerNo), 'wb'))
    print("=> Dictionary Learnt and Saved.\n")
    
    print("=> Learning Weights for Layer {}.".format(layerNo))
    code = sparse_encode(X, mbdl.components_)
    pickle.dump(code, open('../weights/weights_airplane_{}.sav'.format(layerNo), 'wb'))
    print("=> Weights Learnt and Saved.\n")
    
    newX = np.dot(code, mbdl.components_)
    image = np.reshape(newX[imageToDisplay], (32, 32))
    plt.imshow(image)
    plt.savefig('../output/train_airplane_{}.png'.format(layerNo))
    return newX

# To get the reconstruced image, multiply code with dictionary and take any row. 
# Reshappen it to get the image.
def main():
    images = getData(5000)
    images = train(images, 900, 233, 1)
    images = train(images, 900, 233, 2)
    images = train(images, 900, 233, 3)
    #images = train(images, 900, 4, 4)

if __name__ == '__main__':
    main()

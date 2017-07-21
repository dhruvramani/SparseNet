import pickle
import numpy as np
import scipy.io as sio
from sklearn.decomposition import sparse_encode, MiniBatchDictionaryLearning

'''
Directories : 
    ./data/    -  Input data files
    ./weights/ -  Weights saved into this
    ./output/  -  Output saved into this
'''

noFiles = 8
def getData(batchSize):
    global noFiles
    data = np.zeros((batchSize * noFiles, 650250))
    print("=> Generating Data.")
    for i in range(1, noFiles):
        filename = "./data/Feature_train_{}.mat".format(i)
        file = sio.loadmat(filename)
        features = np.array(file['Feat'])
        for j in range(batchSize):
            dat = features[:, j][0]             
            data[j + j * batchSize ''' - 3 * batchSize'''  , : ] = np.ndarray.flatten(dat)
    print("=> Data Ready.")
    return data

'''
# TODO : Test function to be modified when test-data is recieved.
def test(filePath, batchSize, noLayers):
    img = # TODO : Modify
    X = np.ndarray.flatten(img)
    for i in range(1, noLayers+1):
        mbdl = pickle.load(open('./weights/dictionary_mat_{}.sav'.format(i), 'rb'))
        print("=> Learning Weights for Test-Layer {}.".format(i))
        code = sparse_encode(X, mbdl.components_)
        newX = np.dot(code, mbdl.components_)
        pickle.dump(code, open('./weights/weights_mat_{}.sav'.format(i), 'wb'))
        X = newX
        pickle.dump(newX, open('./output/test_mat_{}'.format(i), 'wb'))
        print("=> Weights learnt and Numpy Array Saved for Test-Layer {}.".format(i))
'''

def train(X, noComp, layerNo):
    print("=> Learning Dictionary for Layer {}.".format(layerNo))
    mbdl = MiniBatchDictionaryLearning(noComp)
    mbdl.fit(X)
    pickle.dump(mbdl, open('./weights/dictionary_mat_{}.sav'.format(layerNo), 'wb'))
    print("=> Dictionary Learnt and Saved.\n")    
    print("=> Learning Weights for Layer {}.".format(layerNo))
    code = sparse_encode(X, mbdl.components_)
    pickle.dump(code, open('./weights/weights_mat_{}.sav'.format(layerNo), 'wb'))
    print("=> Weights Learnt and Saved.\n")
    newX = np.dot(code, mbdl.components_)
    return newX

# To get the reconstruced image, multiply code with dictionary and take any row. 
# Reshappen it to get the image.
def main():
    images = getData(10)
    images = train(images, 900, 1)
    images = train(images, 900, 2)
    images = train(images, 900, 3)

if __name__ == '__main__':
    main()

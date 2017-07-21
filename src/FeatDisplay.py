import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def displayData():
    a = sio.loadmat('Feat.mat')['Feat']
    for j in range(10):
        b = a[:, j]
        c = b[0]
        for i in range(51):
            plt.imshow(c[:, :, i])
            plt.savefig('foo/{}_{}.png'.format(j,i))

if __name__ == '__main__':
    displayData()
# Sparse Net 
A scikit-learn based implementation fo Sparse-Net, an unsupervised architecture, which helps in feature detection. It consists of multiple layers of sparse-encoding which envloves breaking the features into two parts, the weight and the dictionary. The dictionary represents the complex structures of the features (like edges etc.) and the weight indicates wether a particular feature in the dictionary exists in the current sample or not. The weight matrix is sparse.

## Installation
Install matplotlib and scikit-learn from their official website before running.

## Running
```shell
cd ./src
python3 SparseNet.py
```
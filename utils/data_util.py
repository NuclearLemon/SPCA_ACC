import os
import scipy.io as scio
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler


"""
Due to the size of the Mnist dataset exceeding the limitations of GitHub, 
we have provided the get_mnist function to automatically generate MNIST.mat file
"""
D_COIL20, D_DIGITS, D_USPS, D_OLI, D_YALE, D_MNIST = 'Coil20', 'Digits', 'USPS', 'Olivetti', 'Yale64', 'MNIST'
D_BASEHOCK, D_PCMAC, D_RELATHE = 'BASEHOCK', 'PCMAC', 'RELATHE'
dataset_list = [
    D_COIL20,
    D_DIGITS,
    D_USPS,
    D_OLI,
    D_YALE,
    D_BASEHOCK,
    D_MNIST,
]

"""
number of classes for different datasets.
"""
dataset_dict = {
    D_COIL20: 20,
    D_DIGITS: 10,
    D_USPS: 10,
    D_OLI: 40,
    D_YALE: 15,
    D_MNIST: 10,
    D_BASEHOCK: 2,
    D_PCMAC: 2,
    D_RELATHE: 2,
}


def get_dataset_name_list():
    return dataset_dict.keys()


def get_mat(path, feature='feature', label='label'):
    """read mat dataset"""
    data = scio.loadmat(path)
    images, labels = data[feature], data[label]
    labels = labels.reshape(-1)
    return images, labels


def to_mat(image_data, image_label, path, name='data'):
    data_dict = {
        'feature': image_data,
        'label': image_label,
    }
    path_full = os.path.join(path, name + '.mat')
    scio.savemat(path_full, data_dict)
    print(path_full + ' already saved')


def get_data(dir, name=D_COIL20):
    """return dataset (n * m)"""
    path = os.path.join(dir, name + '.mat')
    data, label = get_mat(path=path)
    return data, label


def get_dataset_info(dir, name):
    """get dataset info"""
    data, label = get_data(dir, name=name)
    n_components = dataset_dict.get(name)
    return data, label, n_components


def get_mnist(path):
    print('Downloading Mnist dataset')
    data_o, label = fetch_openml('mnist_784', version=1, return_X_y=True, parser='auto')
    data = data_o.values
    print('Downloaded')
    data = StandardScaler().fit_transform(data)
    to_mat(data, label, path, D_MNIST)
    return data, label

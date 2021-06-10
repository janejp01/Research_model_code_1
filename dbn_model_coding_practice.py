from abc import ABCMeta, abstractmethod
import numpy as np
from scipy.stats import truncnorm
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, RegressorMixin


class ActivationFunction(object):
    #class for bastract activation function
    __metaclass__= ABCMeta
    @abstractmethod()
    def function(self,x):
        return
    @abstractmethod()
    def prime(self,x):
        return



class sigmoidActivationFunction(ActivationFunction):
    @classmethod
    def function(cls,x):

        return 1/(1.0+ np.exp(-x))
    @classmethod
    def prime(cls,x):
        return x*(1-x)
class ReLUActivationFunction(ActivationFunction):
    @classmethod
    def function(cls,x):
        return np.maximum(np.zeros(x.shape,x))
    @classmethod
    def prime(cls,x):
        return (x>0).astype(int)

class TanhActivationFunction(ActivationFunction):
    @classmethod
    def function(cls, x):
        return np.tanh(x)
    @classmethod
    def prime(cls,x):
        return 1- x*x

def batch_generator(batch_size, data, labels=None):
    """
        Generates batches of samples
        :param data: array-like, shape = (n_samples, n_features)
        :param labels: array-like, shape = (n_samples, )
        :return:
        """
    n_batches = int (np.ceil(len(data)/float(batch_size)))
    idx = np.random.permutation(len(data))
    data_shuffled = data[idx]
    if lables is not None:
        labels_shuffled = lables[idx]
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        if labels is not None:
            yield data_shuffled[start:end,:], labels_shuffled[start:end]
        else:
            yield data_shuffled[start:end, :]
def to_categorical(labels, num_classes):
    new_labels = np.zeros([len(labels), num_classes])
    label_to_idx_map, idx_to_label_map = dict(), dict()
    idx = 0
    for i ,label in enumerate(labels):
        if label not in label_to_idx_map:
            label_to_idx_map[lable] = idx
            idx_to_label_map[idx] = label
            idx += 1
        new_labels[i][label_to_idx_map[label]] == 1

    return new_labels, label_to_idx_map, idx_to_label_map



class BaseModel(object):
    def save(self, save_path):
        import  pickle
        with open(save_path, 'wb') as fp:
            pickle.dump(self.fp)
    @classmethod
    def load(cls, load_path):
        import pickle
        with open(load_path, 'rb') as fp:
            return pickle.load(fp)


class BinaryRBM(BaseEstimator, TransformerMixin, BaseModel):
    # This class implements a Binary Restricted Boltzmann machine.
    def __init__(self,
                 n_hidden_units =100,
                 activation_function = 'sigmoid',
                 optimization_algorithm = 'sgd',
                 learning_rate = 1e-3,
                 n_epochs = 10,
                 contrastive_divergence_iter =1,
                 batch_size = 32,
                 verbose =True,
                 ):
        self.n_hidden_units =n_hidden_units
        self.n_hidden_units = n_hidden_units
        self.activation_function = activation_function
        self.optimization_algorithm = optimization_algorithm
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.contrastive_divergence_iter = contrastive_divergence_iter
        self.batch_size = batch_size
        self.verbose = verbose

    def fit(self, X):
        # initialize RBM parameters
        self.n_visible_units = X.shape[1]
        if self.activation_function == 'sigmoic':
            self.W = np.random.rand(self.n_hidden_units, self.n_visible_units) / np.sqrt(self.n_visible_units)
            self.c = np.random.rand(self.n_hidden_units) / np.sqrt(self.n_visible_units)
            self.b = np.random.rand(self.n_visible_units) / np.sqrt(self.n_visible_units)
            self._activation_function_class = SigmoidActivationFunction

        elif self.activation_function == 'relu':
            self.W = truncnorm.rvs(-0.2,0.2, size=[self.n_hidden_units, self.n_visible_units]) / np.sqrt(self.n_visible_units)



import numpy as np 
import matplotlib.pyplot as plt 

class Perceptron(object):

    """ Perceptron Object

        Attributes
        ----------

        eta: Float
            Learning Rate
        
        max_iter: int
            Maximum number of epochs

        random_state: int
            Random Seen for reproducable results

    
    """

    def __init__(self, eta=0.01, max_iter=25, random_state=42, id=0):
        
        self.eta = eta
        self.max_iter = max_iter
        self.random_state = random_state
        self.id = id

    def __repr__(self):
        return f"Perceptron Object: {self.id}"

    def fit(self, X, y, func='step'):

        # Ensure the labels match the instances and if not
        # stop the program. 
        
        assert len(X) == len(y)

        random_gen = np.random.RandomState(self.random_state)

        # random.normal - The probability density function of the
        # normal distribution. This function draws a random sample
        # from a Guassian distribution. A mean of 0 and a standard
        # deviation of 1. 

        # params
        # loc: Mean of distribution
        # scale: sd
        # size: output shape
        #  The shape here reflects the input feature columns plus
        #  1 for the bias. 

        self.w_ = random_gen.normal(loc=0.0, scale=0.01,
                                size=1 + X.shape[1])

        self.errors_ = []

        for _ in range(self.max_iter):
            errors = 0
            for xi, target in zip(X, y):
                if func == 'sig':
                    update = self.eta * (target - self.predict(xi, func='sig'))
                else:
                    update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def fit_batch(self, X, y, func='step'):

        assert len(X) == len(y)

        random_gen = np.random.RandomState(self.random_state)

        self.w_ = random_gen.normal(loc=0.0, scale=0.01,
                                    size=1 + X.shape[1])

        self.errors_ = []

        for _ in range(self.max_iter):

            weight_update = np.zeros(len(self.w_))
            errors = 0
            for xi, target in zip(X, y):
                if func == 'sig':
                    update = self.eta * (target - self.predict(xi, func='sig'))
                else:
                    update = self.eta * (target - self.predict(xi))
                weight_update[1:] += update * xi
                weight_update[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)

            self.w_[1:] += np.mean(weight_update[1:]) 
            self.w_[0] += np.mean(weight_update[0])
        return self

    def dot_product(self, X):

        # Used to classify by returning the dot product of and instances
        # feature values with the weights  + the bias. 

        return np.dot(X, self.w_[1:]) + self.w_[0]

    def sigmoid_activation(self, X):

        return 1/(1 + np.exp(-np.dot(X, self.w_[1:]) + self.w_[0]))

    def predict(self, X, func='step'):
        
        # np.where returns elements chosen from x depending
        # on some condition. 

        # Here, I use it to return a prediction, but first the dot product 
        # is returned from the .dot_product() function, returning that instances
        # vectors values multiplied by the weights + the bias. 

        # If this value is greater than or equal to 0 return 1, else return.

        if func == 'sig':
            return self.sigmoid_activation(X)
        else:
            return np.where(self.dot_product(X) >= 0.0, 1, 0)

    def convert_prediction(self, pred_list):

        DEC_BOUND = 0.8
        pred = []
        for p in pred_list:
            if p > DEC_BOUND:
                pred.append(1)
            else:
                pred.append(0)
        return pred

    def retn_prediction_list(self, X, func='step'):

        if func == 'sig':
            pred_list = [self.predict(x, func='sig') for x in X]
            # for p in pred_list:
            #     p = self.convert_prediction(p)
        else:
            pred_list = [self.predict(x) for x in X]
        
        pred_array = np.array(pred_list)
        return pred_array

    def weight_matrix(self):

        pixels = self.w_[1:]
        pixels = pixels.reshape(28, 28)
        plt.title("Perceptron Classifier (7's): Weight Matrix")
        plt.imshow(pixels, cmap='Greys')
        plt.show()

        

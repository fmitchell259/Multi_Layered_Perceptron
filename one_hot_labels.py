"""
    A wee function that will return a numpy
    array with the labels one hot encoded to
    a specific value. 

"""
import numpy as np

def one_hot_label(labels, target):

    """ One Hot Encodes based on a target

        Parameters
        ----------

        labels: 1d Numpy array of labels.
        
        target: Target value to encode, will
                be labelled as 1 and the rest
                as 0.

        Returns
        -------
    
        numpy array: Encoded 1d numpy array.

    """


    return np.where(labels == target, 1, 0)




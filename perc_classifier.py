from perceptron import Perceptron
from one_hot_labels import one_hot_label
from visualise_numbers import visualise_number, visualise_number_prediction
from sklearn import metrics
import numpy as np 


class MultiLayerPerceptron(object):

    """ Multi Layered Perceptron

        Object that holds multiple perceptrons in a 
        dictionary and trains each one on a singular
        number from the MNIST dataset. 
        
        Attributes
        ----------
        
        num_inputs: int
            Number of single perceptrons to build and
            train.
        
        X_train: 2d numpy array
            Instances and features of training set.
            
        y_train: 1d numpy array
            Instance labels
    
    
    """

    def __init__(self, num_inputs, X_train, y_train):

        self.num_inputs = num_inputs
        self.X_train = X_train
        self.y_train = y_train
        self.inputs = {}

    def __repr__(self):
        return "MLP Object"

    def fit(self):
        
        # For a user specified number of inputs, create
        # perceptrons and add them to the input dictionary.
        
        # Train each perceptron on every number in the MNIST
        # dataset. 

        for _ in range(self.num_inputs):

            y_train_ = one_hot_label(self.y_train, _)
            perc = Perceptron(id=_)
            perc.fit(self.X_train, y_train_)
            self.inputs[_] = perc

        return self

    def predict(self, X):
        
        # Iterate over the input dictionary and make
        # predictions on some instance. 
        
        # Save those predictions of a list and choose
        # the index that has a 1 and not a 0. This index
        # reflects the MLP's prediction. 

        pred_list = []
        dp = 0
        for num, model in self.inputs.items():
            pred = model.predict(X)
            pred_list.append(pred)
        
        try:
            num = pred_list.index(1)
        except:
            num = False
        return num


def main():

    # Part 1. Complete the Perceptron.
    # --------------------------------

    # The program begins by creating a perceptron
    # to classify only 7's.

    for _i in range(1):

        # Variable number to predict saved as soon as 
        # the loop starts

        NUMBER_TO_PREDICT = 7

        # Quickly load data from the .npz file. 

        mnist_data = np.load("mnist_scaled.npz")

            # Cool, so using the load_mnist script I now have
            # the data as an easily accessible .npz file, no
            # massive load times. Hoorah!
            # 
            # So using the object as a dictionary I just grab the
            # the data that is required.  

        X_train = mnist_data['X_train']
        X_test = mnist_data['X_test']

        y_train = mnist_data['y_train']
        y_test = mnist_data['y_test']

        print(X_train.shape)

            # I've got my wee one hot encoding function working
            # so lets create training sets for the number 7. 

        y_train_seven = one_hot_label(y_train, NUMBER_TO_PREDICT)
        y_test_seven = one_hot_label(y_test, NUMBER_TO_PREDICT)

            # Use my wee visualise function to confirm the one hot 
            # encoding has worked. I know 15 is a 7

        for _ in range(3):

            visualise_number(X_train[_], y_train[_])

            # The plot and label both correspond to a seven, so
            # lets check my one hot encoding. 

        print(y_train_seven[15])


            # Good, the terminal print out confirms the one hot encoding

            # Now lets instanitaite a classifier and train it. 

        seven_clf = Perceptron(id=NUMBER_TO_PREDICT)

        print(f"[+] Training Classifier on {seven_clf.id}'s...")


        seven_clf.fit(X_train, y_train_seven)

            # Now I test on the first number, which is a 7

        pred_seven = seven_clf.predict(X_test[0])

            # Great, my perceptron, predicts well, now lets
            # iterate over the whole dataframe and make
            # predictions yo!

        full_pred_list = seven_clf.retn_prediction_list(X_test)

        print(f"\n\tPerceptron Accuracy Predicting {seven_clf.id}'s: {metrics.accuracy_score(full_pred_list, y_test_seven) * 100}%")

        # Part 2. Implement Batch Learning
        # --------------------------------

        # Now I use the newly created .batch_fit() method to train the another
        # classifier to predict 7's.

        batch_perc = Perceptron()

        batch_perc.fit(X_train, y_train_seven)

        print("\n[+] Batch Training Classifier on 7's...\n")

        batch_full_list = batch_perc.retn_prediction_list(X_test)
        print(f"\tBatch Perceptron Accuracy Predicting {NUMBER_TO_PREDICT}'s: {metrics.accuracy_score(batch_full_list, y_test_seven) * 100}%\n")


    # Part 3. Use Multiple Nodes to Classify every Digit.
    # --------------------------------------------------

    # First I instantiate an instance of my Multi Layer Perceptron
    # object. I pass this object the full training set plus the labels.

    print('[+] Implementing Multi Layer Perceptron\n')

    mlp = MultiLayerPerceptron(10, X_train, y_train)
    
    # Using the in-built fit method, the MLP will create and train
    # 10 individual perceptrons and store them inside a dictionary.
    
    mlp.fit()

    print("[+] Testing Accuracy of Singular Models in MLP...\n")

    # The following snippet simply tests each of the perceptrons
    # held by the MLP, using the full test set, and prints the
    # accuracy of each one indivdually. 

    for num, model in mlp.inputs.items():

        full_pred = model.retn_prediction_list(X_test)
        y_ = one_hot_label(y_test, num)
        print(f"\t[+] Perceptron {model.id} Accuracy: {metrics.accuracy_score(full_pred, y_) * 100}%\n")

    # Next up I test the MLP by asking it to make predictions over the
    # whoel testing set, and print the accuracy. 

    print(f"[+] Testing MLP Predict Method\n")
    mlp_pred = []
    for _ in range(len(y_test)):

        p = mlp.predict(X_test[_])
        mlp_pred.append(float(p))

    print(f"\tMLP Accuracy: {metrics.accuracy_score(mlp_pred, y_test) * 100}%\n")


    # Here I print out the first twenty predictions along with their actual
    # labels. 

    print(f"[+] Predicting First Twenty Test Instances...\n")

    for _ in range(20):

        print(f"\tTesting Instance {_}.\n")
        print(f"\tActual Number: {y_test[_]}")
        p = mlp.predict(X_test[_])
        print(f"\tPredicted Number: {p}\n")
        print("----------------------------")
    
    print('\n\n')

    # Part 4. Implement A Sigmoid Activation
    # --------------------------------------

    # By simply adding a parameter to the methods in the perceptron
    # object (set by default to 'step') I can make the model train 
    # and predict using the SIgmoid Activation function. 

    print("[+] Testing Single Perceptron Trained on Sigmoid")

    sigmoid_perceptron = Perceptron()
    sigmoid_perceptron.fit(X_train, y_train_seven, func='sig')
    sigmoid_full_pred = sigmoid_perceptron.retn_prediction_list(X_test, func='sig')
    print(f"[+] Sigmoid Activation Accuracy Predicting 7's: {metrics.accuracy_score(sigmoid_full_pred, y_test_seven) * 100}%\n")

    # Part 5. Print the data to the screen and the weights. 

    # First I iterate over the first five instances in the test set,
    # print them to screen and use a seven classifer to make a 
    # prediction on them. 

    for _ in range(10):

        pred = seven_clf.predict(X_test[_])
        visualise_number_prediction(X_test[_], y_test[_], pred, 7)


    # # And finally I print the weights of the 7 classifier to the screen

    seven_clf.weight_matrix()


main()
    



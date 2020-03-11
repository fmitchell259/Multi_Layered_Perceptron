
import matplotlib.pyplot as plt 
import time

def visualise_number(instance, label):

    img = instance.reshape(28, 28)
    plt.title(f"MNIST Instance: {int(label[0])}")
    plt.imshow(img, cmap='plasma')
    plt.show(block=False)
    plt.pause(2)
    plt.close('all')


def visualise_number_prediction(test_instance, test_label, prediction, predicting):

    if prediction == 0:
        p = "NO"
    else:
        p = "YES"
    img = test_instance.reshape(28, 28)
    plt.title(f"MNIST Test Instance: {int(test_label[0])}")
    plt.xlabel(f"IS THIS A {predicting}: {p}!")
    plt.imshow(img, cmap='plasma')
    plt.show(block=False)
    plt.pause(2)
    plt.close('all')





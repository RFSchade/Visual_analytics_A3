#================================================================#
#=============> VGG16 feature extraction CNN Model <=============#
#================================================================#

#=====> Import modules
# os
import os

# tf tools
import tensorflow as tf

# image processsing
from tensorflow.keras.preprocessing.image import (load_img,
                                                  img_to_array,
                                                  ImageDataGenerator)
# VGG16 model
from tensorflow.keras.applications.vgg16 import (preprocess_input,
                                                 decode_predictions,
                                                 VGG16)
# cifar10 data - 32x32
from tensorflow.keras.datasets import cifar10

# layers
from tensorflow.keras.layers import (Flatten, 
                                     Dense, 
                                     Dropout, 
                                     BatchNormalization)
# generic model object
from tensorflow.keras.models import Model

# optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import SGD

#scikit-learn
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

# for plotting
import numpy as np
import matplotlib.pyplot as plt

#=====> Define functions
# > Prepate data
def prep_data():
    # Print info
    print("[INFO] loading data...")
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    # Normalize data 
    X_train = X_train/255
    X_test = X_test/255
    # Create one-hot encodings (Still not sure what this does)
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.fit_transform(y_test)
    # Initialize label names
    label_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    # Print info
    print("[INFO] Data loaded")
    
    return (X_train, y_train), (X_test, y_test), label_names

# > Create model
def create_model():
    # Print info 
    print("[INFO] Initializing model")
    
    # > Initialize model 
    model = VGG16(include_top = False, # Do not include classifier!
                  pooling = "avg", # Pooling the final layer  
                  input_shape = (32, 32, 3)) # Defineing input shape
    # Disable training on convolutional layers
    for layer in model.layers:
        layer.trainable = False
        
    # > Add layers 
    # The second pair of closed brackets is the input 
    flat1 = Flatten()(model.layers[-1].output) # create a flatten layer from the output for the last layer of the model
    class1 = Dense(128, activation='relu')(flat1)
    output = Dense(10, activation='softmax')(class1)
    # Adding everything together
    model = Model(inputs = model.inputs, 
                  outputs = output)
    
    # Print info
    print("[INFO] Compiling model")
    # Slowing down the model's learning to avoid overfitting
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=1000,
        decay_rate=0.9)

    sgd = SGD(learning_rate=lr_schedule)
    # Compiling model
    model.compile(optimizer=sgd,
             loss="categorical_crossentropy", # binary_crossentropy for binary categories 
             metrics=["accuracy"])
    # Print info
    print("[INFO] Model compiled!")
    print("[INFO] Model summary:")
    model.summary()
    
    return model
    

# > Evaluate model
def evaluate(model, X_test, y_test, label_names):
    # evaluate network
    predictions = model.predict(X_test, batch_size=32)
    # print classification report
    print(predictions[0])
    # print classification report
    report = classification_report(y_test.argmax(axis=1), 
                                   predictions.argmax(axis=1), 
                                   target_names=label_names)
    # Print metrics
    print(report)
    # Save metrics
    outpath = os.path.join("output", "classification_report.txt")
    with open(outpath, "w") as f:
        f.write(report)
        
# > Plot history
def plot_history(H, epochs):
    plt.style.use("seaborn-colorblind")

    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, epochs), H.history["val_loss"], label="val_loss", linestyle=":")
    plt.title("Loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(np.arange(0, epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, epochs), H.history["val_accuracy"], label="val_acc", linestyle=":")
    plt.title("Accuracy curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.legend()
    # Saving image
    plt.savefig(os.path.join("output", "history_img.png"))

#=====> Define main()
def main():
    (X_train, y_train), (X_test, y_test), label_names = prep_data()
    model = create_model()
    history = model.fit(X_train, y_train,
             validation_data = (X_test, y_test), # Was there a way to split up the validation data further?
             batch_size = 128, # two to the power of something to optimize memory
             epochs = 10, # Should I change this later? Perhaps make an argument that allows the user to specify so I don't have to think about it?
             verbose = 1) # Tell me what is happening 
    evaluate(model, X_test, y_test, label_names)
    plot_history(history, 10)

# Run main() function from terminal only
if __name__ == "__main__":
    main()

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


INIT_LR = 1e-4   #learnig rate

EPOCHS = 20


#it is a parameter denotes the subset size of your training sample which is going to be used in order to train the network during its learning process
BATCH_SIZ = 32
# EPOCHS and BATCH_SIZ:
 


DIRECTORY = r"F:\Face-Mask-Detection\dataset"                #location of dataset
CATEGORIES = ["with_mask", "without_mask"]


print("[INFO] loading images...")                           #output screen

# it traces obj that represents the underlying  pixels data of an area of the image
img_data = []                              
img_labels = []                    # labels of mask and without mask

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)      #directory and category ko join kiya
    for img in os.listdir(path):                       # category matlab mask and without mask
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(100, 100))   # load from karence
        image = img_to_array(image)
        image = preprocess_input(image)

        img_data.append(image)        #append means to add an item to the end of theexisting item
        img_labels.append(category)


lb = LabelBinarizer()            #convert with mask and without mask in binary
labels = lb.fit_transform(img_labels)                   
labels = to_categorical(labels)
# np is a python lib for working with array
data = np.array(img_data, dtype="float64")     #make array
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split (data, labels,
                                 test_size=0.20, stratify=labels,
                                 random_state=28)

# load the MobileNetV2 network 

baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(224, 224, 3)))            # CNN model with mobilenet(FASTER than CNN)
#mobilenetv2: you can load pretrained version of the network on more than million images from imagenet dataset

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)  #for activaction
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

 
model = Model(inputs=baseModel.input, outputs=headModel)
 
for layer in baseModel.layers:
    layer.trainable = False

# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# construct the training image generator for data augmentation
print("[INFO] generating images by changing its properties...")
Img_Gen = ImageDataGenerator(
    rotation_range=25,                       #img processing
                                             #creating many img from one img with diff angle
    zoom_range=0.25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# train the head of the network
print("[INFO] training head...")
H = model.fit(
    Img_Gen.flow(trainX, trainY, batch_size=BATCH_SIZ),
    steps_per_epoch=len(trainX) // BATCH_SIZ,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BATCH_SIZ,
    epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] predicting model...")
predIdxs = model.predict(testX, batch_size=BATCH_SIZ)


predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
                            target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure(figsize=(12,8))
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs...")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="center right")
plt.savefig("accuracy plot.png")



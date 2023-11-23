# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import os
import winsound
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'       #python lib for war

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 2.0, (224, 224), (104.0, 177.0, 123.0))   #img prosess in nn 224 is size of img  104--pixels
     # blob -- numpy array
    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # initialize our list


    faces = []
    locs = []   #location XY ractangle
    preds = []     #acuurecy %

    # loop hoga
    for i in range(0, detections.shape[2]):
        

        confidence = detections[0, 0, i, 2]

         
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (100, 100))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=30)

    
    return (locs, preds)

# load our serialized face detector model from disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)                         # CV2 DNN DEEP NEURAL NETWORK

# load the face mask detector model 
maskNet = load_model("mask_detector.model")       #model that we develope

# initialize the video stream
print("[INFO] started capturing video stream...")
vs = VideoStream(src=0).start()                            # src is camera 

# loop over the frames from the video stream
while True:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # frame for video stream 
    #  maximum width 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=600)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        if mask > withoutMask:
            label = "Mask"
        else:
            label = "Wear Mask"
            winsound.PlaySound('alert.wav', winsound.SND_LOOP)
            file_path = "F:/Face-Mask-Detection/without_mask_detected/"
            cv2.imwrite(file_path+str(now.replace(':','-'))+".jpg", frame)

        color = (0, 255, 0)  if label == "Mask" else (0, 0, 255)  #green  and red89

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)   #percentage of mask

        # display the label and bounding box rectangle on the output frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
    cv2.putText(frame, "Press 'e' to Exit", (10, 430), cv2.FONT_HERSHEY_COMPLEX_SMALL, .75, (13, 204, 253), 1)
    #  current DateTime on frame
    cv2.putText(frame, str(now), (150, 40), cv2.FONT_HERSHEY_SIMPLEX, .75, (255, 255, 255), 1, cv2.LINE_AA)
    # show the output frame
    cv2.imshow("Mask Detector", frame)
    # Capture video after each 1 milliseconds
    key = cv2.waitKey(1)

    # if the `e` key was pressed, break from the loop
    if key == ord("e"):
        break

cv2.destroyAllWindows()
vs.stop()

import tensorflow as tf
from tensorflow import keras
from keras import layers
# from tensorflow.keras import layers
import cv2
import os
import matplotlib.pyplot as plt
import random
import numpy as np
Datadirectory = "prain"
Classes = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
for category in Classes:
    path = os.path.join(Datadirectory, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img))
        # plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
        # plt.show()
img_size = 224
new_array = cv2.resize(img_array, (img_size, img_size))
plt.imshow(cv2.cvtColor(new_array, cv2.COLOR_BGR2RGB))
plt.show()
training_data = []

def create_training_Data():
    for category in Classes:
        path = os.path.join(Datadirectory, category)
        class_num = Classes.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (img_size, img_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
create_training_Data()
random.shuffle(training_data)
#np.array(training_data)
X, y = [], []
for features, label in training_data:
    X.append(features)
    y.append(label)
X = np.array(X).reshape(-1, img_size, img_size, 3)
X = X/255.0
Y = np.array(y)

# Deep learning model for training - transfer learning

model = tf.keras.applications.MobileNetV2()
base_input = model.layers[0].input
base_output = model.layers[-2].output
final_output = layers.Dense(128)(base_output)                # Adding new layers after global pooling layer output
final_output = layers.Activation("relu")(final_output)       # activation function
final_output = layers.Dense(64)(final_output)
final_output = layers.Activation("relu")(final_output)
final_output = layers.Dense(7, activation="softmax")(final_output)       # 7 classes

new_model = keras.Model(inputs=base_input, outputs=final_output)
new_model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
new_model.fit(X, Y, epochs=5)
new_model.save("exp_det_model.h5")

"""cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, currentFrame = cap.read()
    if ret:"""


"""frame = cv2.imread("/Users/DELL/PycharmProjects/expressionDetection1/Dataset/train/angry/Training_397587.jpg")
plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(gray, 1.1, 4)
for (x, y, w, h) in faces:
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = frame[y:y+h, x:x+w]
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    facess = faceCascade.detectMultiScale(roi_gray)
    if len(facess) == 0:
        print("Face not detected")
    else:
        for (ex, ey, ew, eh) in facess:
            face_roi = roi_color((ey+eh), (ex+ew))
            ey = ey+eh
            ex = ex+ew

plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
final_image = cv2.resize(face_roi, (224, 224))
final_image = np.expand_dims(final_image, axis=0)
final_image = final_image/255.0
Predictions = new_model.predict(final_image)
print(Predictions[0])"""

# Realtime

path = r"C:\Users\Lenovo\PycharmProjects\pythonProject\haarcascade_frontalface_default.xml"
font_scale = 1.5
font = cv2.FONT_HERSHEY_PLAIN
rectangle_bgr = (255, 255, 255)
img = np.zeros((500, 500))
text = "some text in a box"
(text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
text_offset_x = 10
text_offset_y = img.shape[0] - 25
box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_width - 2))
cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("cannot open webcam")

while True:
    ret, frame = cap.read()
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        facess = faceCascade.detectMultiScale(roi_gray)
        if len(facess) == 0:
            print("Face not detected")
        else:
            for (ex, ey, ew, eh) in facess:
                face_roi = roi_color[ey: ey + eh, ex: ex + ew]

    final_image = cv2.resize(face_roi, (224, 224))
    final_image = np.expand_dims(final_image, axis=0)
    final_image = final_image/255.0

    font = cv2.FONT_HERSHEY_SIMPLEX
    Predictions = new_model.predict(final_image)
    font_scale = 1.5

    """if np.argmax(Predictions) == "angry":
        status = "Angry"
    elif np.argmax(Predictions) == "disgust":
        status = "Disgust"
    elif np.argmax(Predictions) == "fear":
        status = "Fear"
    elif np.argmax(Predictions) == "happy":
        status = "Happy"
    elif np.argmax(Predictions) == "neutral":
        status = "Neutral" """

    x1, y1, w1, h1 = 0, 0, 175, 75
    cv2.rectangle(frame, (x1, x1), (x1 + w1, y1 + h1), (0, 0, 0), -1)
    cv2.putText(frame, np.argmax(Predictions), (x1 + int(w1/10), y1 + int(h1/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, np.argmax(Predictions), (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255))
    """elif np.argmax(Predictions) == "disgust":
        status = "Disgust"""

    cv2.imshow("Face Emotion Recognition", frame)
    if cv2.waitKey(2) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
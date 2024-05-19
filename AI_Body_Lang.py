import cv2 as cv
import mediapipe.python.solutions.drawing_utils
import numpy as np
from mediapipe.python import *
import mediapipe.python.solutions as solutions
import os
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression , RidgeClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle


draw = mediapipe.python.solutions.drawing_utils
stick = mediapipe.python.solutions.holistic  # this is for drawing lines

cap = cv.VideoCapture(0)
#'C:/Users/user/Videos/Captures/vid.mp4'
# ---------------- Knowing number of coordinates of face and pose ---------------- #
num_coordinate = 1000
landmarks = ['class']
for val in range(1, num_coordinate + 1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

""" if you want to over write a new thing and delete the old info --> decomment this code::
# ---------------- Creating a new file with csv---------------- #
with open('coordinates.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)
"""
# ---------------- Reading file by pandas  ---------------- #
df = pd.read_csv('coordinates.csv')  # df --> DataFrame

# ---------------- target:class , features:coordinates ---------------- #
x = df.drop('class', axis=1)  # feature
y = df['class']                     # target = 'y' will contain the values from the 'class' column of 'df'.

# ---------------- split the data into training and testing sets  ---------------- #
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1234)

# ---------------- creating the pipeline  ---------------- #
pipelines = {
    'lr' : make_pipeline(StandardScaler(), LogisticRegression()),
    'rc' : make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf' : make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb' : make_pipeline(StandardScaler(), GradientBoostingClassifier())
}

# ---------------- training step, store values  ---------------- #
fit_model = {}
for alogrithm, pipeline in pipelines.items():
    model = pipeline.fit(x_train, y_train)
    fit_model[alogrithm] = model

for alogrithm, model in fit_model.items():
    yhat = model.predict(x_test)

# ---------------- saving the models in file  ---------------- #
# with open('body_lang.pkl', 'wb') as f:  # wb --> write binary
#     pickle.dump(fit_model['rf'], f)

with open('body_lang.pkl', 'rb') as f:  # wb --> read binary
    model = pickle.load(f)


class_name = 'sad'

# ---------------- Initiate holistic model ---------------- #
with stick.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7) as holistic:

    while True:
        ret, frame = cap.read()

        # ---------------- Change color to RGB  ---------------- #
        image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image.flags.writeable = False ## prevent copying image data, but you can use same image for rendering

        # ---------------- Make Detection ---------------- #
        result = holistic.process(image)

        # ---------------- Recolor image to bgr for rendering ---------------- #
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        image.flags.writeable = True

        # ---------------- Draw face landmark ---------------- #
        draw.draw_landmarks(image, result.face_landmarks, stick.FACEMESH_CONTOURS, landmark_drawing_spec= draw.DrawingSpec(color=(121, 22, 0), circle_radius=1, thickness=1))

        # ---------------- Draw right hand landmark ---------------- #
        draw.draw_landmarks(image, result.right_hand_landmarks, stick.HAND_CONNECTIONS, landmark_drawing_spec= draw.DrawingSpec(color=(123, 22, 0), circle_radius=1, thickness=1))

        # ---------------- Draw left hand landmark ---------------- #
        draw.draw_landmarks(image, result.left_hand_landmarks, stick.HAND_CONNECTIONS, landmark_drawing_spec= draw.DrawingSpec(color=(122, 22, 0), circle_radius=1, thickness=1))

        # ---------------- Draw pose landmark ---------------- #
        draw.draw_landmarks(image, result.pose_landmarks, stick.POSE_CONNECTIONS, landmark_drawing_spec= draw.DrawingSpec(color=(122, 23, 0), circle_radius=1, thickness=1))

        # ---------------- Exporting the coordinates ---------------- #
        try:
            # extract landmarks for pose
            pose = result.pose_landmarks.landmark

            # extract all the coordinates in arrays for pose
            pose_rows = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

            # extract landmarks for face
            face = result.face_landmarks.landmark

            # extract all the coordinates in arrays for face
            face_rows = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

            # concat the rows
            row = pose_rows + face_rows

            # # Append the class
            # row.insert(0, class_name)
            #
            # # Export to CSV
            # with open('coordinates.csv', mode='a', newline='') as f:
            #     csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            #     csv_writer.writerow(row)

            # Make Detections
            x = pd.DataFrame([row])
            body_lang_class = model.predict(x)[0]
            body_lang_prob = model.predict_proba(x)[0]
            # print(body_lang_class, body_lang_prob)

            # Grab ear coordinates
            ear_coords = tuple(np.multiply(
                                np.array(
                                    (result.pose_landmarks.landmark[stick.PoseLandmark.LEFT_EAR].x,
                                     result.pose_landmarks.landmark[stick.PoseLandmark.LEFT_EAR].y))
                            , [640, 480]).astype(int))

            # Drawing and Texting
            cv.rectangle(image, (ear_coords[0], ear_coords[1]+5), (ear_coords[0]+len(body_lang_class)*20, ear_coords[1]-30), (245, 117, 16), -1)
            cv.putText(image, body_lang_class, ear_coords, cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv.LINE_AA)

            # get status box
            cv.rectangle(image, (0, 0), (250, 60), (245, 117, 16), -1)

            # Display Class
            cv.putText(image, 'CLASS'
                        , (95, 12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
            cv.putText(image, body_lang_class.split(' ')[0]
                        , (90, 40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

            # Display Probability
            cv.putText(image, 'PROB'
                        , (15,12), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
            cv.putText(image, str(round(body_lang_prob[np.argmax(body_lang_prob)],2))
                        , (10,40), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)



        except:
            pass

        cv.imshow('frame', image)
        if cv.waitKey(1) == ord('q'):
            break

cv.destroyWindow('frame')




"""
The flatten() method is a function available in NumPy that is used to convert a multi-dimensional array into a one-dimensional array

min_detection_confidence::
It determines how confident the model needs to be in its detection of a landmark before considering it a valid detection.
Increasing this value can help filter out.

min_tracking_confidence::
It determines the minimum confidence required for the model to continue tracking a landmark across consecutive frames.
Increasing this value can make the tracking more robust, as it requires higher confidence to maintain the tracking.

The drop() function in pandas is used to remove specified columns or rows from a DataFrame.
The axis=1 parameter indicates that we are dropping a column (axis=1) instead of a row (axis=0).

x: This represents the input features or independent variables, typically stored in a DataFrame or an array.
y: This represents the target variable or dependent variable, typically stored in a Series or an array.
test_size=0.3: This specifies the proportion of the dataset that should be allocated for testing. 
In this case, 30% of the data will be used for testing, while the remaining 70% will be used for training.
random_state=1234: This parameter is used to set a specific random seed for reproducibility. 
By providing a value, in this case, 1234, the same random splits will be generated each time the code is executed.

Training set: During the training process, the model learns patterns and relationships in the training data to make predictions or perform a specific task.
The training set is typically larger than the testing set to ensure the model has enough data to learn from.

Testing Set: The testing set is a subset of the data that is used to evaluate the performance of a trained machine learning model.
The process of splitting the data into training and testing sets helps evaluate the model's ability to generalize to new, unseen samples. 
By training the model on the training set and evaluating it on the testing set, you can estimate how well the model will perform on future, real-world data.


"""
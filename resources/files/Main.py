from tkinter import Tk, Button, Label, filedialog
import cv2 # pip install opencv-contrib-python
from PIL import Image, ImageTk #pip install Pillow
import mediapipe as mp
from keras_preprocessing.image import img_to_array
from keras.models import load_model
import os
import numpy as np
import json

# Read parameters ###########################################################
with open("parameters.json") as data:
    parameters=json.loads(data.read())

    # Mediapipe
mp_hands=mp.solutions.hands
mp_drawing=mp.solutions.drawing_utils

#Cargamos el modelo y pesos para la predicción
folder='resources/files/'
cnn = load_model(folder+'model.h5')
cnn.load_weights(folder+'weights.h5')
dire_img = os.listdir(folder+'validation')


def streamingVideo():
    global cap
    cap=cv2.VideoCapture(0)
    showVideo()

def showVideo():
    global cap
    ret, frame=cap.read()
    # Getting backgroun image
    background=cv2.imread('resources/images/background.png')
    if ret == True:
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ##############################################
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5
        ) as hands:

            result=hands.process(frame)
            detection=result.multi_hand_landmarks

            if detection is not None:
                for hand_landmarks in detection:
                    # Paintng dots and linies
                    mp_drawing.draw_landmarks(
                    background, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    # Changing color
                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=0, circle_radius=0),
                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=8)
                )

                x1, y1=220, 140 
                x2, y2=420, 340 
                cut=background[y1:y2, x1:x2]
                cut=cv2.cvtColor(cut, cv2.COLOR_BGR2GRAY)
                cut=cv2.resize(cut, (parameters['width'], parameters['height']))
                vector = cnn.predict(np.expand_dims(img_to_array(cut), axis=0))
                r=vector[0]
                state=dire_img[np.argmax(vector[0])]
                faceContL['text']='Letra identificada: {}'.format(state)
                cv2.rectangle(frame, (x1,y1), (x2,y2), (255, 255, 255), 3)
        ##############################################


        i=Image.fromarray(frame)
        img=ImageTk.PhotoImage(image=i)
        videoL.configure(image=img)  
        videoL.image=img
        videoL.after(10, showVideo) 

window=Tk()
window.title('Clasificador de manos (Alfabeto de señas R, S)')
window.state('zoomed')
window.configure(bg='#42a8a1')


# Buton para elegir video
seeStreamingB=Button(window, text='Usar cámara', font=('Times', 24),bg='#3E77B6', command=streamingVideo)
seeStreamingB.grid(row=0, column=1, padx=100, pady=50)


text=Label(window, text='Ubique las señas dentro del recuadro')
text.grid(row=0, column=0)
text.configure(bg='#42a8a1', font=('Times', 24))

# Label para el video
videoL=Label(window)
videoL.grid(row=2, column=0, padx=50)

#Identificador de señas
faceContL=Label(window, text='Letra identificada: ', font=('Times', 24),bg='#3E77B6')
faceContL.grid(row=2, column=1, padx=10)

window.mainloop()
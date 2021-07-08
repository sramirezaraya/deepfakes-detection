import tkinter
from tkinter import *
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import cv2
from keras.models import load_model
import numpy as np
import keras
import tensorflow
import os
from mtcnn import MTCNN

ventana = tkinter.Tk()
ventana.geometry("768x687")
ventana.configure(bg="white")
ventana.title("Sistema Clasificador de Deepfakes")

print(keras.__version__)
print(tensorflow.__version__)

path = "./modelos/"
name_model = "VGG16.h5"
model = load_model(os.path.join(path,name_model))
detector = MTCNN()

# funcion crop

def crop(box,image):
    x0 = box[0]
    y0 = box[1]
    w= box[2]
    h= box[3]
    if x0<0:
      x0=0
    if y0<0:
      y0=0
    if type(image) is np.ndarray:
        if image.size==0:
            pass
    if image is None:
        pass
    image = cv2.resize(image[y0:y0+h , x0:x0+w],(224,224))
    return image


def prediccion(filename):
    m_pred = []
    count = 0
    cap = cv2.VideoCapture(filename)
    while cap.isOpened():
        success, image = cap.read()
        if success:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = detector.detect_faces(image)
            if len(face_locations) > 0:
                for person in face_locations:
                    if person['confidence'] > 0.95:
                        i = 0
                        bounding_box = person['box']
                        keypoints = person['keypoints']
                        confidence = person['confidence']
                        image = np.expand_dims(crop(bounding_box, image), axis=0)
                        PRED = model.predict(image)[0][0]
                        m_pred.append(PRED)
                        i += 1
            count += 150
            #count += int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 20)
            cap.set(1, count)
        else:
            cap.release()
            break
    return video2(filename, np.mean(m_pred))

# esta funcion no crea un nuevo video, sino que solo muestra la prediccion en el video entregado.

def video2(filename, pred):
  cap = cv2.VideoCapture(filename)
  if (cap.isOpened()== False):
    print("Error al abrir el video")

  while(cap.isOpened()):
    success, image = cap.read()
    if success == True:
      face_locations =  detector.detect_faces(image)
      if len(face_locations) > 0:
            for person in face_locations:
              if person['confidence']>0.95:
                  bounding_box = person['box']
                  if pred>=0.5:
                        cv2.rectangle(image,
                            (bounding_box[0], bounding_box[1]),
                            (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                            (0,0,255),
                            2)
                        text = "FAKE" + " - " + str(pred)
                        cv2.putText(image,str(text),(bounding_box[0],bounding_box[1]-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2,cv2.LINE_AA)
                  else:
                    cv2.rectangle(image,
                        (bounding_box[0], bounding_box[1]),
                        (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                        (0,255,0),
                        2)
                    text = "REAL" + " - " + str(pred)
                    cv2.putText(image,str(text),(bounding_box[0],bounding_box[1]-5),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA)
      cv2.imshow('Prediccion Video', image)
      key = cv2.waitKey(1)
      if key & 0xFF == ord('q'):
        break
    else:
        break

def cargar_archivo():
	filename = askopenfilename()
	prediccion(filename)
	

img3 = Image.open("facial.png")
render2 = ImageTk.PhotoImage(img3)
img_3 = Label(ventana, image=render2, bd=0)
img_3.place(x=0,y=0)

img2 = PhotoImage(file="button.png")
b1 = Button(ventana, image= img2, bd=0, command=cargar_archivo)
b1.place(x=260,y=640)

ventana.mainloop()

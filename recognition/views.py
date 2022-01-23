from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from recognition.camera import FaceDetect
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from recognition import extract_embeddings
from recognition import train_model
import cv2,os,urllib.request,pickle
import imutils
from face_recognition import api
# import urllib # python 2
import urllib.request # python 3
import json
import cv2
import os
from django.shortcuts import render
from django.http import HttpResponse
from recognition.forms import FaceRecognitionform
from django.conf import settings
from recognition.models import FaceRecognition
from recognition.models import User, Person, ThiefLocation
from PIL import Image, ImageDraw, ImageFont
from django.shortcuts import render, HttpResponse, redirect
# Create your views here.

def index(request):
	return render(request, 'recognition/index.html')

def home(request):
	return render(request, 'recognition/home.html')

def gen(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
		
def facecam_feed(request):
	return StreamingHttpResponse(gen(FaceDetect()),
					content_type='multipart/x-mixed-replace; boundary=frame')
def Train(request):
    extract_embeddings.embeddings()
    train_model.model_train()
    return render(request, 'recognition/home.html')
    cv2.destroyAllWindows()



protoPath = os.path.sep.join([ "face_detection_model\\deploy.prototxt"])
modelPath = os.path.sep.join(["face_detection_model\\res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    #Tải model của embedding
embedder = cv2.dnn.readNetFromTorch(os.path.join('face_detection_model/openface_nn4.small2.v1.t7'))
# Tải mô hình nhận dạng cùng với bộ nhãn
recognizer = os.path.sep.join(["output\\recognizer.pickle"])
recognizer = pickle.loads(open(recognizer, "rb").read())
le = os.path.sep.join(["output\\le.pickle"])
le = pickle.loads(open(le, "rb").read())
dataset = os.path.sep.join(["dataset"])
user_list = [ f.name for f in os.scandir(dataset) if f.is_dir() ]
def detectImage(request):
    # This is an example of running face recognition on a single image
    # and drawing a box around each person that was identified.

    # Load a sample picture and learn how to recognize it.

    #upload imagerequest.method == 'POST' and
    if request.FILES['image']:
        myfile = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(myfile.name, myfile)
        uploaded_file_url = fs.url(filename)
    # else:

    #     return render(request, 'home.html', context={'uploaded_file_url': uploaded_file_url})
        #person=Person.objects.create(name="Swimoz",user_id="1",address="2020 Nehosho",picture=uploaded_file_url[1:])
        #person.save()

    frame = api.load_image_file(uploaded_file_url[1:])
    frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]
    imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(frame, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)
    detector.setInput(imageBlob)
    detections = detector.forward()     
    for i in range(0, detections.shape[2]): 
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]
            if fW < 20 or fH < 20:
                continue
            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = le.classes_[j]
            # if proba*100 <40:
                # name = "Unknown"
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            with open('attendance.csv','r+') as g:
                attendance = g.read()
                attendance = attendance.split('\n')
                if name not in attendance:
                    attendance.append(name)
                    g.close()
                    g = open('attendance.csv','w')
                    for a in attendance:
                        g.write(a)
                        g.write('\n')
                    g.close()


    cv2.imshow('frame',frame)
    cv2.waitKey(0)
    return render(request, 'recognition/home.html')
    cv2.destroyAllWindows()






    
    





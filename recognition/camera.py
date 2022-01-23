# ------------------------
#CAPSTONE PROJECT:
#ROBUST STUDENT ATTENDANCE CHECK SYSTEM BASED ON A DEEP LEARNING MODEL
# SUPERVISOR: PH.D NGUYEN DINH VINH
# STUDENT’S NAME: NGUYEN DONG BANG
# STUDENT’S NAME: LUU DUC HOA
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import cv2,os,urllib.request,pickle
import numpy as np
from django.conf import settings
from recognition import extract_embeddings
from recognition import train_model
  # Tải bộ dò khuôn mặt, deploy prototxt và caffemodel để tạo các layer và các tham số của các layer
protoPath = os.path.sep.join([ "face_detection_model/deploy.prototxt"])
modelPath = os.path.sep.join(["face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"])
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    #Tải model của embedding
embedder = cv2.dnn.readNetFromTorch(os.path.join('face_detection_model/openface_nn4.small2.v1.t7'))
# Tải mô hình nhận dạng cùng với bộ nhãn
recognizer = os.path.sep.join(["output/recognizer.pickle"])
recognizer = pickle.loads(open(recognizer, "rb").read())
le = os.path.sep.join(["output/le.pickle"])
le = pickle.loads(open(le, "rb").read())
dataset = os.path.sep.join(["dataset"])
user_list = [ f.name for f in os.scandir(dataset) if f.is_dir() ]

class FaceDetect(object):
	def __init__(self):
		# extract_embeddings.embeddings()
		# train_model.model_train()
#Khởi tạo viideo, khi start video thì sẽ chạy hàm detect_face
		self.vs = VideoStream(src=0).start()
		# start the FPS throughput estimator
		self.fps = FPS().start()

	def __del__(self):
		cv2.destroyAllWindows()

	def get_frame(self):
#Lấy khung hình từ video
		frame = self.vs.read()
		frame = cv2.flip(frame,1)

#Thay đổi kích thước khung hình để có chiều rộng 600px và sau đó lấy hình ảnh
		frame = imutils.resize(frame, width=600)
		(h, w) = frame.shape[:2]

#Tạo đốm màu từ ảnh
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(frame, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)

#Áp dụng tính năng dò khuôn mặt để chuyển khuôn mặt về vector, khuon mặt trong ảnh đầu vào
		detector.setInput(imageBlob)
		detections = detector.forward()
#Tạo vòng lặp phát hiện khuôn mặt trong ảnh
		for i in range(0, detections.shape[2]):
#Tính xác suất và show xác suất
			confidence = detections[0, 0, i, 2]
#Lọc ra các xác xuất thấp hơn 50%
			if confidence > 0.5:
#tinh tọa độ của bouding box
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
#Trích xuất vùng dữ liệu quan tâm(ROI)
				face = frame[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]
#Kiểm tra chiều rộng và chiều cao đủ lớn để tạo ra khuôn mặt
				if fW < 20 or fH < 20:
					continue
#Tạo đốm màu cho ROI, thông qua khuon mặt nhận dạng tao ra vector
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()
#thực hiện phân loại khuôn mặt theo các label
				preds = recognizer.predict_proba(vec)[0]
				j = np.argmax(preds)
				proba = preds[j]
				name = le.classes_[j]
				# if proba*100 <40:
				# 	name = "Unknown"
#Xuất ra bouding box và tên khuôn mặt
				text = "{}: {:.2f}%".format(name, proba * 100)
				y = startY - 10 if startY - 10 > 10 else startY + 10
				cv2.rectangle(frame, (startX, startY), (endX, endY),
					(0, 0, 255), 2)
				cv2.putText(frame, text, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)	
 				#markAttendance(name)
				# with open('attendance.csv','r+') as g:
				# 	attendance = g.read()
				# 	attendance = attendance.split('\n')
				# 	if name not in attendance:
				# 		attendance.append(name)
				# 		g.close()
				# 		g = open('attendance.csv','w')
				# 		for a in attendance:
				# 			g.write(a)
				# 			g.write('\n')
				# 		g.close()






#Cập nhật bộ đếm FPS
		self.fps.update()
		ret, jpeg = cv2.imencode('.jpg', frame)
		return jpeg.tobytes()




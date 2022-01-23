# ------------------------
#CAPSTONE PROJECT:
#ROBUST STUDENT ATTENDANCE CHECK SYSTEM BASED ON A DEEP LEARNING MODEL
# SUPERVISOR: PH.D NGUYEN DINH VINH
# STUDENT’S NAME: NGUYEN DONG BANG
# STUDENT’S NAME: LUU DUC HOA
from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os
from django.conf import settings

#------------------------------------------------------#
#---------PHÂN TÁCH ẢNH THÀNH CÁC ĐIỂM NHÚNG-----------#
#------------------------------------------------------#
def embeddings():
	# load our serialized face detector from disk
	print("[INFO] loading face detector...")
    # Tạo bộ dò khuôn mặt, deploy prototxt và caffemodel để tạo các layer và các tham số của các layer
	protoPath = os.path.sep.join(["face_detection_model/deploy.prototxt"])
	modelPath = os.path.sep.join(["face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"])
	detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    #Tải model của embedding
	print("[INFO] loading face recognizer...")
	embedder = cv2.dnn.readNetFromTorch(os.path.join('face_detection_model/openface_nn4.small2.v1.t7'))

    # Lấy danh sách các đường dẫn đến các ảnh
	print("[INFO] quantifying faces...")
	dataset = os.path.sep.join(["dataset"])
	imagePaths = list(paths.list_images(dataset))
    # Khởi tạo danh sách các điểm nhúng và danh sách các label
	knownEmbeddings = []
	knownNames = []

    # Khởi tạo các khuôn mặt được xữ lý
	total = 0

    # Lặp lại qua từng đường dẫn đến ảnh
	for (i, imagePath) in enumerate(imagePaths):
      # Trích xuất tên của ảnh
		print("[INFO] processing image {}/{}".format(i + 1,
			len(imagePaths)))
		name = imagePath.split(os.path.sep)[-2]
        # Tải hình ảnh, thay đổi kích thước để có chiều rộng là 600px, giũ nguyên tỷ lệ khuôn hình và sau đó lấy kích thước của ảnh
		image = cv2.imread(imagePath)
		image = imutils.resize(image, width=600)
		(h, w) = image.shape[:2]

        # Tạo đốm màu từ hình ảnh
		imageBlob = cv2.dnn.blobFromImage(
			cv2.resize(image, (300, 300)), 1.0, (300, 300),
			(104.0, 177.0, 123.0), swapRB=False, crop=False)

        # Áp dụng do khuôn mặt dựa trên opencv để khoanh vùng các khuôn mặt trong hình ảnh đầu vào
		detector.setInput(imageBlob)
		detections = detector.forward()


        # Đảm bảo rằng có ít nhất một khuôn mặt được phát hiện
		if len(detections) > 0:
            # Đưa ra giả định rằng mổi hình ảnh có một khuôn mặt được phát hiện và có một độ chính xác cao nhất
			i = np.argmax(detections[0, 0, :, 2])
			confidence = detections[0, 0, i, 2]
            # Đảm bảo rằng khuôn mặt được phát hiện có độ chính xác cao hơn một mức độ chỉ định
			if confidence > 0.5:
                # Tính toán tọa độ của bounding box
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
                # Trích xuất ROI của khuôn mặt và lấy độ chính xác của khuôn mặt
				face = image[startY:endY, startX:endX]
				(fH, fW) = face.shape[:2]
                # Đảm bảo chiều rộng và chiều cao của khuôn mặt là đủ lớn
				if fW < 20 or fH < 20:
					continue
                # Tạo đốm màu cho ROI của khuôn mặt, chuyển đốm màu đó qua mô hình nhúng của khuôn mặt để có được 128-d vector
				faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
					(96, 96), (0, 0, 0), swapRB=True, crop=False)
				embedder.setInput(faceBlob)
				vec = embedder.forward()
                # Thêm tên người vào danh sách các tên và danh sách các điểm nhúng
				knownNames.append(name)
				knownEmbeddings.append(vec.flatten())
				total += 1

    # Các điểm nhúng và tên được lưu vào một file .pickle
	print("[INFO] serializing {} encodings...".format(total))
	data = {"embeddings": knownEmbeddings, "names": knownNames}
	embeddings = os.path.sep.join(["output/embeddings.pickle"])
	f = open(embeddings, "wb")
	f.write(pickle.dumps(data))
	f.close()
#--------------------------------------------------------------------------#

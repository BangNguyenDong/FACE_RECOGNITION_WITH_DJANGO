# ------------------------
#CAPSTONE PROJECT:
#ROBUST STUDENT ATTENDANCE CHECK SYSTEM BASED ON A DEEP LEARNING MODEL
# SUPERVISOR: PH.D NGUYEN DINH VINH
# STUDENT’S NAME: NGUYEN DONG BANG
# STUDENT’S NAME: LUU DUC HOA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle,os
from django.conf import settings
#------------------------------------------------------#
#---------------------TRAIN MODEL----------------------#
#------------------------------------------------------#
def model_train():
# Tải các điểm nhúng và tên của các khuôn mặt được lưu trong file .pickle
	print("[INFO] loading face embeddings...")
	embeddings = os.path.sep.join(["output/embeddings.pickle"])
	data = pickle.loads(open(embeddings, "rb").read())


   # Mã hóa các điểm nhúng và tên của các khuôn mặt
	print("[INFO] encoding labels...")
	le = LabelEncoder()
	labels = le.fit_transform(data["names"])

    # Đào tạo mô hình nhúng của các khuôn mặt
	print("[INFO] training model...")
	recognizer = SVC(C=1.0, kernel="linear", probability=True)
	recognizer.fit(data["embeddings"], labels)

    # Các điểm nhúng và tên của các khuôn mặt được lưu vào một file .pickle
	recognizers = os.path.sep.join(["output/recognizer.pickle"])
	f = open(recognizers, "wb")
	f.write(pickle.dumps(recognizer))
	f.close()

  # Các nhãn của các khuôn mặt được lưu vào một file .pickle
	le_pickle = os.path.sep.join(["output/le.pickle"])
	f = open(le_pickle, "wb")
	f.write(pickle.dumps(le))
	f.close()
    #------------------------------------------------------#

import cv2

thres = 0.5 # Threshold to detect object

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
cap.set(3,1280)
cap.set(4,720)
cap.set(10,70)
w=1280
h=720
classNames= []
classFile = "coco.names"
with open(classFile,"rt") as f:
	classNames = [line.rstrip() for line in f]#f.read().rstrip("n").split("n")

configPath ="E:/M64/tello/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
#"ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "E:/M64/tello/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)
def display(img):
	cv2.putText(img,
				"MA64:obstacle", (30, 10), 2, 1, (255, 255, 255), 2)
	cv2.line(img,
             (int(w/3),0),(int(w/3),h),
             (255,0,230),
             3)
	cv2.line(img,
             (2*int(w / 3), 0), (2*int(w / 3), h),
             (255, 0, 230),
             3)
	cv2.line(img,
             (0,int(h / 3)), (w,int(h / 3)),
             (255, 0, 230),
             3)
	cv2.line(img,
             (0,2*int(h / 3)), (w,2*int(h / 3)),
             (255, 0, 230),
             3)

while True:
	success,img = cap.read()
	classIds, confs, bbox = net.detect(img,confThreshold=thres)
	print(classIds,bbox)

	if len(classIds) != 0:
		for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
			cv2.rectangle(img,box,color=(0,255,0),thickness=2)
			cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
			cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
			cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
			cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

			if classNames[classId - 1].lower() == "person":
				cx = box[0] + box[2] // 2
				cy = box[1] + box[3] // 2
				area = box[2] * box[3]
				cv2.circle(img, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

				i = img.copy()

				if cx < int(w / 3):
					cv2.circle(i, (cx+130, cy), 10, (0, 255, 0), cv2.FILLED)
				if cx > (2 * int(w / 3)):
					cv2.circle(i, (cx-130, cy), 10, (0, 255, 0), cv2.FILLED)
				if cy > (2 * int(h / 3)):
					cv2.circle(i, (cx, cy-130), 10, (0, 255, 0), cv2.FILLED)
				if cx in range((int(w / 3)), (2 * (int(w / 3)))) and cy in range((int(h / 3)), (2 * (int(h / 3)))):
					cv2.rectangle(i,(cx,cy),(cx+30,cy+30),(0,235,210),3,cv2.FILLED)


				display(i)
				cv2.imshow("disp",i)
				cv2.imshow("Output",img)
	cv2.waitKey(1)
cv2.destroyAllWindows()
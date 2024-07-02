from ultralytics import YOLO
import cv2
import cvzone
import math

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone',
              'Safety Vest', 'machinery', 'vehicle']


cap = cv2.VideoCapture('ppe1.mp4')
model = YOLO('best.pt')
myColor = (0, 0, 255)


while True:
    success, img = cap.read()

    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1

            conf = math.ceil(box.conf[0]*100) / 100

            cls = int(box.cls[0])
            currentClass = classNames[cls]


            if conf>0.5:
                if currentClass =='NO-Hardhat' or currentClass =='NO-Safety Vest' or currentClass == "NO-Mask":
                    myColor = (0, 0,255)
                elif currentClass =='Hardhat' or currentClass =='Safety Vest' or currentClass == "Mask":
                    myColor =(0,255,0)
                else:
                    myColor = (255, 0, 0)

                cv2.rectangle(img, (x1, y1), (x2, y2), myColor, 3) 
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1,colorB=myColor,
                                   colorT=(255,255,255),colorR=myColor, offset=5)
            

    cv2.imshow('VIDEO', img)
    cv2.waitKey(1)
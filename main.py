from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]


cap = cv2.VideoCapture('Cars.mp4')

model = YOLO('yolov8l.pt')

mask = cv2.imread('mask.png')

# Tracking
tracker = Sort(max_age=20,
               min_hits=3,
               iou_threshold=0.3)

limits = [400, 297, 673, 297]
total_count = list()

while True:
    success, img = cap.read()

    imgregion = cv2.bitwise_and(img, mask)

    img_graphics = cv2.imread(r'D:\Study\Course Archive\CV_youtube\Murtaza_Advanced\Car Counter\graphics.png', cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, img_graphics)

    results = model(imgregion, stream=True)

    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2-x1, y2-y1
            

            conf = math.ceil(box.conf[0]*100) / 100
            cls = int(box.cls[0])
            current_class = classNames[cls]

            if (current_class == "car" or current_class == "truck" or current_class == "bus") \
            and conf >0.3:
                # cvzone.putTextRect(img, f'{current_class} {conf}', (max(0,x1), max(y1,35)), scale=0.6, 
                #                    thickness=1, offset=3)
                # cvzone.cornerRect(img, (x1,y1,w,h), l=9, rt=5)
                current_array = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, current_array))

            
    results_tracker = tracker.update(detections)

    # cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]),
    #          (0,0,255), 5)

    for result in results_tracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)

        print(result)
        cvzone.cornerRect(img, (x1,y1,w,h), l=9, rt=2, colorR=(255,0,255))
        cvzone.putTextRect(img, f'{id}', (max(0,x1), max(y1,35)), scale=0.6, 
                                   thickness=1, offset=3)
        
        cx, cy = x1 + w//2, y1 + h//2
        cv2.circle(img, (cx,cy), 5, (0,255,0), cv2.FILLED)

        if limits[0] < cx < limits[2] and limits[1]-30 < cy < limits[1]+30:
            if total_count.count(id) == 0:
                total_count.append(id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]),
                    (0,255,0), 5)
    
    # cvzone.putTextRect(img, f'Count: {len(total_count)}', (50,50))
    cv2.putText(img, str(len(total_count)), (255,100), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0), 8)        

    cv2.imshow('VIDEO', img)
    # cv2.imshow('MASK', imgregion)
    cv2.waitKey(1)
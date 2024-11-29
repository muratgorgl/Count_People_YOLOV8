import cv2
import time 
from ultralytics import YOLO
import os 

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("video.mp4")

pTime = 0

p1 = (200, 720)
p2 = (1000, 300) 
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    print(results)
    human_count = 0
 
    for result in results[0].boxes:  
        x1, y1, x2, y2 = result.xyxy[0].tolist()  
        conf = result.conf[0].item()  
        cls = result.cls[0].item()  
        
        if int(cls) == 0: 
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.putText(frame, f"Human {conf:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)

            if x1 > p1[0] and y1 < p1[1] and x2 < p2[0] and y2 > p2[1]:
                human_count+=1 

 
    overlay = frame.copy()
    opacity = 0.2
    cv2.rectangle(overlay, (200, 720), (1000, 300), (255, 0, 0), thickness=cv2.FILLED)
    cv2.addWeighted(overlay, opacity, frame, 1 - opacity, 0, frame)  

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame, "FPS: " + str(int(fps)),(10,50),cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
    cv2.putText(frame, "Human Conunt: " + str(int(human_count)),(10,100),cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)

    cv2.imshow("frame",frame)
    print(frame.shape)

    if cv2.waitKey(20) & 0xFF == ord("q"): break


cap.release()    
cv2.destroyAllWindows()




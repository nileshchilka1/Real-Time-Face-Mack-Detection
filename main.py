import cv2
import numpy as np
from fdet import RetinaFace
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("location", help="Location of a video")
parser.add_argument("--threshold", help="(Default=0.5) Threshold value whether it is face",type=float,default=0.5)
args = parser.parse_args()

detector = RetinaFace(backbone='RESNET50',threshold=args.threshold)

cap = cv2.VideoCapture(args.location)

frame_width = int(cap.get(3)) 
frame_height = int(cap.get(4)) 
   
size = (frame_width, frame_height) 

output = cv2.VideoWriter('output.avi',  
                         cv2.VideoWriter_fourcc(*'MJPG'), 
                         10, size) 


count=0
while(1):
            
    ret,image = cap.read()
    
    if not ret:
        break
                    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
    result = detector.detect(image)
                
    if count == 0:
        from tensorflow.keras import models
        model = models.load_model('face_mask')
        count=1
    boxes = []
    if len(result) > 0:
                
        for i in range(len(result)):
            boxes.append(result[i]['box'])
                        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    
        for (x, y, w, h) in boxes:
                
            face = image[max(y-5,0):y+h+5,max(x-5,0):x+w+5]
            face = cv2.resize(face, (128, 128),interpolation=cv2.INTER_CUBIC)
                    
            face = face.reshape(1,128,128,3)
            face = face / 255
            target =  model.predict(face)[0][0]
    
            if target > 0.5:
                cv2.rectangle(image, (x,y), (x+w, y+h), (0, 0, 225), 2)
            else:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
    else:
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    cv2.imshow('webcam',image)
    
    output.write(image)

    if cv2.waitKey(1) == 13:
        break
                    
                    
cap.release()
output.release() 
cv2.destroyAllWindows()
          
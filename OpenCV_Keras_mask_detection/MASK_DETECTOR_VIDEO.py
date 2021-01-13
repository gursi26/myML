import cv2, os, sys
from keras.models import load_model
import numpy as np

#%%

current_path = os.path.dirname(os.path.abspath(sys.argv[0]))

mask_model = load_model(current_path + '/mask_modelR3E15.h5')

prototxtpath = current_path + '/facenet/deploy.prototxt'
caffemodelpath = current_path + '/facenet/res10_300x300_ssd_iter_140000.caffemodel'
facenet = cv2.dnn.readNet(prototxtpath, caffemodelpath)

#%%

video_path = str(input('Enter path to video. Leave empty for camera feed : '))

if video_path == '' :
    video_path = int(0)
    
vid = cv2.VideoCapture(video_path)
initial_state = False
try : 
    while True :
        success, test_img = vid.read()
        initial_shape = test_img.shape
        (h, w) = initial_shape[:2]
        
        blob = cv2.dnn.blobFromImage(test_img, 1.0, (128, 128), (104.0, 177.0, 123.0))
        facenet.setInput(blob)
        detections = facenet.forward()
        
        for i in range(0, detections.shape[2]):
            conf = detections[0,0,i,2].copy()
            if conf > 0.7 : 

                box = detections[0, 0, i, 3:7] * np.array([w,h,w,h])
                (startx, starty, endx, endy) = box.astype('int')
        
        try : 
            
            model_input = cv2.resize(test_img[starty:endy, startx:endx].copy(), (128,128))

            temp_1 = model_input[:, :, 0].copy()
            temp_2 = model_input[:, :, 2].copy()

            model_input[:, :, 2] = temp_1
            model_input[:, :, 0] = temp_2

            initial_state = True
            model_input = model_input.reshape((1, 128, 128, 3))/255.0

            pred = mask_model.predict(model_input)[0]

            if pred[0] > pred[1] : 
                cv2.rectangle(test_img, (startx,starty), (endx, endy), (0,255,0), 3)
                cv2.putText(test_img, 'mask %.2f' %pred[0], (startx, starty), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,100), 2)
            else : 
                cv2.rectangle(test_img, (startx,starty), (endx, endy), (255,0,0), 3)
                cv2.putText(test_img, 'no mask %.2f' %pred[1], (startx,starty), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,100), 2)

            cv2.imshow('Output', test_img)
            
        except NameError : 
            pass

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
        
except KeyboardInterrupt :
    vid.release()
    
#%%
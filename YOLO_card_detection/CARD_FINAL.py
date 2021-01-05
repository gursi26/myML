import cv2
import os, sys

current_path = os.path.dirname(os.path.abspath(sys.argv[0]))

which_yolo = int(input('Enter 0 for yolo and 1 for yolo-tiny : '))

if which_yolo == 0 :
    cfgpath = current_path + '/weights_configs/yolo_test.cfg'
    weightspath = current_path + '/weights_configs/yolov3_custom_final.weights'
elif which_yolo == 1 :
    cfgpath = current_path + '/weights_configs/yolov3-tiny_test.cfg'
    weightspath = current_path + '/weights_configs/yolov3-tiny_custom_final.weights'
else :
    print('Not a valid input')

net = cv2.dnn.readNetFromDarknet(cfgpath, weightspath)

classes = ["card"]

cap = cv2.VideoCapture(0)
try : 
    while True :
        _, img = cap.read()
        img = cv2.resize(img, (1280,720))
        height, width = img.shape[:2]

        blob = cv2.dnn.blobFromImage(img, 1/255.0, (416,416), (0,0,0), swapRB = True, crop= False)
        net.setInput(blob)
        preds = net.forward()

        for var1 in range(0, len(preds)):
            confs = preds[var1][4]
            if confs.max() > 0.01 :
                
                confidence = round(confs.max(), 3)
                prediction = classes[confs.argmax()]

                x_center = int(preds[var1][0] * width)
                y_center = int(preds[var1][1] * height)

                w = int(preds[var1][2] * width)
                h = int(preds[var1][3] * height)

                x_start = int(x_center - w/2)
                y_start = int(y_center - h/2)

                x_end = int(x_start + w)
                y_end = int(y_start + h)

                cv2.rectangle(img, (x_start,y_start), (x_end,y_end), (0,255,0), 2)
                cv2.putText(img, prediction + ' ' + str(confidence), (x_start, y_start), 
                            cv2.FONT_HERSHEY_PLAIN, 1.5, (255,255,255), 2)

        cv2.imshow('Output', img)
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
except KeyboardInterrupt :
    pass
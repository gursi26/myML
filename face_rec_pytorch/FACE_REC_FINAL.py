import cv2 , os, sys
import torch
from torch import nn
from torchvision import transforms
import numpy as np
from PIL import Image

class NET(nn.Module):

    def __init__(self, out1=32, out2=64, n1=64, n2=32):
        super(NET,self).__init__()

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=out1, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 1)
        )

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=out1, out_channels=out2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 4, stride = 2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(out2 * 30 * 30, n1),
            nn.BatchNorm1d(n1),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(n1, n2),
            nn.BatchNorm1d(n2),
            nn.ReLU()
        )

        self.output = nn.Linear(n2,2)

    def forward(self,x):
        x = self.convblock1(x)
        x = self.convblock2(x)

        x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x)
        return x

print('Imports complete')
current_path = os.path.dirname(os.path.abspath(sys.argv[0]))

transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()])

print('-' * 50)
model_path = input('Enter path of model. Leave empty for default model : ')
print('-' * 50)

if model_path == '':
    model_path = '/Users/gursi/Desktop/ML/myML/face_rec_pytorch/facerec_model.pt'

mask_model = torch.load(model_path)
mask_model.eval()

prototxtpath = current_path + '/face_rec_models/deploy.prototxt'
caffemodelpath = current_path + '/face_rec_models/res10_300x300_ssd_iter_140000.caffemodel'

facenet = cv2.dnn.readNet(prototxtpath, caffemodelpath)

print('-' * 50)
video_path = input("Enter path to video. Leave empty for live camera feed : ")
print('-' * 50)

if video_path == '' :
    video_path = int(0)

vid = cv2.VideoCapture(video_path)

try : 
    while True :
        success, test_img = vid.read()
        initial_shape = test_img.shape
        (h, w) = initial_shape[:2]
        
        blob = cv2.dnn.blobFromImage(test_img, 1.0, (256, 256), (104.0, 177.0, 123.0))
        facenet.setInput(blob)
        detections = facenet.forward()
        
        conf = detections[0,0,:,2]
        msk = [conf > 0.7]
        detections = detections.reshape((detections.shape[2],detections.shape[3]))
        detections = detections[msk[0]]
        
        faces = []
        coordinates = []
        for i in range(0, len(detections)):
            
            box = detections[i, 3:7] * np.array([w,h,w,h])
            
            (startx, starty, endx, endy) = box.astype('int')
            coordinates.append([startx, starty, endx, endy])
            
            model_input = test_img[starty:endy, startx:endx].copy()
            
            try : 
                model_input = cv2.resize(model_input.copy(), (256,256))
            except :
                pass
            
            temp_1 = model_input[:, :, 0].copy()
            temp_2 = model_input[:, :, 2].copy()

            model_input[:, :, 2] = temp_1
            model_input[:, :, 0] = temp_2
            
            if model_input.size != 0 :
                model_input = model_input.reshape((1, 256, 256, 3))/255.0
              
            faces.append(model_input)
        
        if len(faces) != 0 :
            
            for some_var in range(0, len(coordinates)):
                try : 
                    le_input = torch.from_numpy(faces[some_var][0])
                    le_input = transforms(le_input)
                    le_input = le_input.view(1, le_input.shape[0], le_input.shape[1], le_input.shape[2])
                    pred = mask_model.forward(le_input)[0]
                except UnboundLocalError :
                    pass
                x,y,ex,ey = coordinates[some_var]

                if pred[0] > pred[1] : 
                    cv2.rectangle(test_img, (x,y), (ex, ey), (0,255,0), 3)
                    cv2.putText(test_img, 'gursi %.4f' %pred[0], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,100), 2)
                else : 
                    cv2.rectangle(test_img, (x,y), (ex, ey), (0,0,255), 3)
                    cv2.putText(test_img, 'no gursi %.4f' %pred[1], (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,100), 2)

        cv2.imshow('Output', test_img)
        
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
        
except KeyboardInterrupt :
    vid.release()
    
#%%


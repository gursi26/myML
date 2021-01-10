from torch.utils.data import Dataset
from keras.preprocessing.image import load_image, img_to_array
import os

'''
Dataset class that loads in images. use with torch.utils.data.DataLoader for loading images

data_object = Data(img_folder, target_size = (128,128), transform = None)

data_object[index] returns image and label at that index. Use labelmap to convert label to string.

- img_folder as a pathname string
- target_size as a tuple with two elements
- Use tranforms from torchvision.transforms

self.labelmap for labelmap
self.len for len
self.img_names for a list of all image names
self.classes for classes


'''
class Data(Dataset):

    def __init__(self, img_folder, target_size = (128,128), transform = None):
        self.classes = os.listdir(img_folder)

        try : 
            self.classes.remove('.DS_Store')
        except ValueError :
            pass

        self.class_path = [os.path.join(img_folder, i) for i in self.classes]
        self.img_names1 = []

        for var2 in self.class_path :
            img_names_in_class = os.listdir(var2)
            self.img_names1.append(img_names_in_class)

        self.img_names = []
        for var3 in self.img_names1 :
            for element in var3 :
                self.img_names.append(element)

        try : 
            self.img_names.remove('.DS_Store')
        except ValueError :
            pass

        self.len = len(self.img_names)
        self.target_size = target_size
        self.transform = transform
        
        self.labelmap = {}
        for var1 in range(len(self.classes)) :
            self.labelmap[var1] = self.classes[var1]

    def __getitem__(self, index):
        
        for i in range(len(self.class_path)) :
            temp_list = os.listdir(self.class_path[i])
            if self.img_names[index] in temp_list :
                break

        image = img_to_array(load_img(os.path.join(self.class_path[i], self.img_names[index]), target_size = self.target_size))
        image = image/255.0
        label = i

        if self.transform :
            image = self.transform(image)

        return image,label

    def __len__(self):
        return self.len
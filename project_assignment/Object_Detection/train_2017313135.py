import os
import sys
import matplotlib.pyplot as plt
import xml.etree.ElementTree as Et
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import shutil

from xml.etree.ElementTree import Element, ElementTree
from PIL import Image
from PIL import ImageDraw
from torchvision.models import resnet152
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
#from torch.utils.tensorboard import SummaryWriter
from test_2017313135 import *

lrate = 0.001
SEED = 100
b_size = 50
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img_path = "./JPEGImages"
ann_path = "./Annotations"
img_resize_path = "./JPEGImages_resized"

_classes = []
_idx2size = []
_idx2map = []
_class2idx = {}

transform = transforms.Compose([
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

epochs = 20
lambda_hasobj = 5
lambda_noobj = 0.5

def setRandom():
    os.environ['PYTHONHASHSEED'] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

def parse_Xml(idx2file):
    print("\n\nNow start parsing xml files")
    print("--------------------------------------------------------\n\n")

    for i, file in enumerate(idx2file):

        xml = open(os.path.join(ann_path, file), "r")
        tree = Et.parse(xml)
        root = tree.getroot()

        img_size = root.find("size")
        width = img_size.find("width").text
        height = img_size.find("height").text
        channels = img_size.find("depth").text

        size=[int(width), int(height), int(channels)]
        _idx2size.append(size)
        
        if ((i % 1000) == 0):    
            print("image {}, image size: {}".format(i, _idx2size[i]))

            
        objects= root.findall("object")

        _map = {}
        for j, obj in enumerate(objects):
            _class = obj.find("name").text

            if _class not in _classes:
                _classes.append(_class)

            box = obj.find("bndbox")
            xmin = box.find("xmin").text
            ymin = box.find("ymin").text
            xmax = box.find("xmax").text
            ymax = box.find("ymax").text

            box_loc = [int(float(xmin)), int(float(ymin)), int(float(xmax)), int(float(ymax))]
            _map[_class] = box_loc

        _idx2map.append(_map)

        if ((i%1000) == 0):
            for key, box in _idx2map[i].items():
                print("    class : {}, box_size: {}".format(key, box))

def resize_images(image_dir, output_dir, size):
    print("\n\nStart Image Resizing")
    print("--------------------------------------------------------\n\n")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    images = os.listdir(image_dir)
    num_images = len(images)
    
    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = img.resize(size, Image.ANTIALIAS)
                img.save(os.path.join(output_dir, image), img.format)
            if (i + 1) % 1000 == 0:
                print("[{}/{}] Image Resized".format(i+1, num_images))

                
def resize_boxinfo():
    print("\n\nStart Box offset transforming")
    print("--------------------------------------------------------\n\n")
    
    for i in range(len(_idx2size)):
        width_ratio = 224 / _idx2size[i][0]
        height_ratio = 224 / _idx2size[i][1]

        if((i%1000) == 0):
            print("image {}, image size: {}".format(i, _idx2size[i]))    
        
        for key, box in _idx2map[i].items():

            _boxinfo = box
            _boxinfo[0] = int(_boxinfo[0] * width_ratio)
            _boxinfo[1] = int(_boxinfo[1] * height_ratio)
            _boxinfo[2] = int(_boxinfo[2] * width_ratio)
            _boxinfo[3] = int(_boxinfo[3] * height_ratio)
    
        if ((i%1000) == 0):
            for key, box in _idx2map[i].items():
                print("    class : {}, box_size: {}".format(key, box))


def makeValRandom():
    
    idxList = list(range(len(totalDS)))
    np.random.shuffle(idxList)

    split = int(np.floor(0.7 * len(totalDS)))
    split2 = int(np.floor(0.9 * len(totalDS)))
    trainIdx, validIdx, testIdx = idxList[:split], idxList[split:split2], idxList[split2:]

    train_sampler = SubsetRandomSampler(trainIdx)
    valid_sampler = SubsetRandomSampler(validIdx)
    test_sampler = SubsetRandomSampler(testIdx)
    
    train_loader = DataLoader(
        dataset=totalDS,
        batch_size=b_size,
        num_workers=0,
        sampler=train_sampler
    )
    
    val_loader = DataLoader(
        dataset=totalDS,
        batch_size=b_size,
        num_workers=0,
        sampler=valid_sampler
    )
    
    test_loader = DataLoader(
        dataset=totalDS,
        batch_size=b_size,
        num_workers=0,
        sampler=test_sampler
    )
    
    return train_loader, val_loader, test_loader




class customDS(data.Dataset):
    def __init__(self, root, idx2map, idx2files, _class2idx, transform=None):
        self.root = root
        self.idx2map = idx2map
        self._class2idx = _class2idx
        self.idx2files = idx2files
        self.y_data = []
        self.transform = transform
        
        for i in range(len(idx2map)):
            np_label = np.zeros((7, 7, 25), dtype=np.float32)
            
            for key, box in idx2map[i].items():
                xoff = int((box[0]+box[2])/2)
                yoff = int((box[1]+box[3])/2)
                width = box[2] - box[0]
                height = box[3] - box[1]
                
                x_idx = xoff//32
                y_idx = yoff//32
                
                x_ratio = (xoff%32)/32
                y_ratio = (yoff%32)/32
                
                
                class_onehot = np.zeros(20, dtype=np.float32)
                class_onehot[_class2idx[key]] = 1
                
                np_label[x_idx][y_idx] = np.concatenate((np.array([1, x_ratio, y_ratio, width/224, height/224]), class_onehot))
                label = torch.from_numpy(np_label)
            self.y_data.append(label)
                    
    def __getitem__(self, index):
        target = self.y_data[index]
        x_file = self.idx2files[index].split(".")[0]+".jpg"
        
        image = Image.open(os.path.join(self.root, x_file)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
            
        return image, target
        
    def __len__(self):
        return len(self.idx2files)



class Obj_Detect(nn.Module):
    def __init__(self):
        super().__init__()
        pretrained = resnet152(pretrained=True)
        layers = list(pretrained.children())[:-1]
        self.resnet = nn.Sequential(*layers)
        
        self.fc1 = nn.Sequential(
            nn.Linear(pretrained.fc.in_features, 4096),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.LeakyReLU(),
            nn.Dropout(0.3)
        )
        
        self.fc3 = nn.Linear(2048, 7 * 7 * 25)
        
    def forward(self, img):
        with torch.no_grad():
            features = self.resnet(img)
        features = features.view(features.size(0), -1)
        
        out = self.fc1(features)
        out = self.fc2(out)
        out = self.fc3(out)
        
        out = out.reshape((-1, 7, 7, 25))
        out[:, :, :, 0] = torch.sigmoid(out[:, :, :, 0])
        out[:, :, :, 5:] = torch.sigmoid(out[:, :, :, 5:])
        
        return out


def loss_func(predict, target):
    
    batch_size = target.size(0)
    
    hasobj_predict = predict[:, :, :, 0]
    x_predict = predict[:, :, :, 1]
    y_predict = predict[:, :, :, 2]
    w_predict = predict[:, :, :, 3]
    h_predict = predict[:, :, :, 4]
    class_predict = predict[:, :, :, 5:]
    
    hasobj_target = target[:, :, :, 0]
    x_target = target[:, :, :, 1]
    y_target = target[:, :, :, 2]
    w_target = target[:, :, :, 3]
    h_target = target[:, :, :, 4]
    class_target = target[:, :, :, 5:]

    nohasobj_target = torch.neg(torch.add(hasobj_target, -1))
    
    box_loss = lambda_hasobj * torch.sum(hasobj_target * (torch.pow(x_predict - x_target, 2) + torch.pow(y_predict - y_target, 2)))
    
    size_loss = lambda_hasobj * torch.sum(hasobj_target * (torch.pow(w_predict - torch.sqrt(w_target), 2) + torch.pow(h_predict - torch.sqrt(h_target), 2)))
    
    hasobj_loss = torch.sum(hasobj_target * torch.pow(hasobj_predict - hasobj_target, 2))
    
    noobj_loss = lambda_noobj * torch.sum(nohasobj_target * torch.pow(hasobj_predict - hasobj_target, 2))
    
    cls_map = hasobj_target.unsqueeze(-1)

    for i in range(19):
        cls_map = torch.cat((cls_map, hasobj_target.unsqueeze(-1)), 3)
    
    class_loss = torch.sum(cls_map * torch.pow(class_predict - class_target, 2))
    
    
    
    total_loss = box_loss + size_loss + class_loss + noobj_loss + hasobj_loss
    total_loss = total_loss / batch_size
    
    return total_loss, hasobj_loss / batch_size
    

#if using tensorboard, have to use writer parameter
def train(train_loader, val_loader):
    print("\n start learning")
    print("------------------------------------------------\n\n")
    
    
    model = Obj_Detect().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lrate)
    
    
    total_step = len(train_loader)
    valid_step = len(val_loader)

    for epoch in range(epochs):

        trn_loss = 0
        trn_batch = 0
        
        model.train()
        for i, (img, target) in enumerate(train_loader):

            img, target = img.to(device), target.to(device)
            trn_batch += target.size(0)
            
            hypothesis = model(img)
            loss, _ = loss_func(hypothesis, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            trn_loss += loss.item()
            if((i % 10) == 0):
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, i + 1, total_step, loss.item()))

        val_loss = 0
        total_conf_score = 0
        val_batch = 0

        print("\nEvaluating validation loss in this epoch")
        model.eval()
        with torch.no_grad():

            for i, (img, target) in enumerate(val_loader):
                img, target = img.to(device), target.to(device)
                val_batch += target.size(0)

                results = model(img)
                loss, confidence = loss_func(results, target)

                val_loss += loss.item()
                total_conf_score += confidence.item()
            print("\n    Valid - total_conf_score_loss: {:.2f} total batchsize : {}".format(total_conf_score, val_batch)) 
            print('      Epoch [{}/{}], Loss: {:.4f}, avg_confidence_score_loss: {:.5f}\n'.format(epoch + 1, epochs, val_loss, total_conf_score/val_batch))
        
##            writer.add_scalar('Loss/train_loss', trn_loss, epoch + 1)
##            writer.add_scalar('Loss/valid_loss', val_loss, epoch + 1)
        
        
##        if(epoch == 4):
##            torch.save(model.state_dict(), './2017313135_권동민1.pt')
##        elif(epoch == 9):
##            torch.save(model.state_dict(), './2017313135_권동민2.pt')
        if(epoch == 14):
            torch.save(model.state_dict(), './model_2017313135.pt')
##        elif(epoch == 19):
##            torch.save(model.state_dict(), './2017313135_권동민4.pt')
        
            
if __name__ == '__main__':
    setRandom()
    
##    writer = SummaryWriter()
    
    _, _, idx2file = next(os.walk(ann_path))
    
    parse_Xml(idx2file)
    
    print("---------------- result parsing ------------------\n")
    print("classes : {}\n".format(_classes))
    print("The number of classes : {}".format(len(_classes)))
    print("The number of total img : {}".format(len(_idx2size)))
    print("\n--------------------------------------------------")
    
    for i in range(len(_classes)):
        _class2idx[_classes[i]] = i

    image_size = [224, 224]

    resize_images(img_path, img_resize_path, image_size)
    resize_boxinfo()

    totalDS = customDS(img_resize_path, _idx2map, idx2file, _class2idx, transform=transform)
    
    print("total images : ")
    print(len(totalDS))
    
    train_loader, val_loader, test_loader = makeValRandom()

    ## for using tensorboard, have to put writer parameter
    train(train_loader, val_loader)
    
##    writer.close()
    test(test_loader)



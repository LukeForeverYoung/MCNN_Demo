import math

import cv2
import sys
from src.network import CrowdCounter,npToTensor
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
class Config():
    def __init__(self):
        self.isDir = False
        self.modelPath = 'model.torch_model'
        self.needShowResult = False
        self.needOutput = False
        self.outResultPath = None
        self.filePath = None
        self.configure(self.read_cmd())

    def configure(self,cmd_dict):
        if os.path.isdir(self.filePath):
            self.isDir=True
        if 'm' in cmd_dict.keys():
            self.modelPath = cmd_dict['m']
        if 's' in cmd_dict.keys():
            self.needShowResult = True
        if 'o' in cmd_dict.keys():
            self.needOutput=True
            self.outResultPath = cmd_dict['o']

    def read_cmd(self):
        self.filePath = sys.argv[1]
        index = 2
        cmd_dict = {}
        while index < len(sys.argv):
            cmd = sys.argv[index].lower()
            if cmd == '-s':
                cmd_dict['s'] = True
            elif cmd == '-o':
                cmd_dict['o'] = sys.argv[index + 1]
                index += 1
            elif cmd == '-m':
                cmd_dict['m'] = sys.argv[index + 1]
                index += 1
            else:
                print('Unexpected Command.')
            index += 1
        return cmd_dict

    def print(self):
        print('Information')
        if self.isDir:
            print('Image Directory Path:',self.filePath)
        else:
            print('Image Path:',self.filePath)
        print('Model Path:',self.modelPath)
        if self.needOutput:
            print('Output Path:',self.outResultPath)
        if self.needShowResult:
            print('Print result at runtime.')
        print('--------')


def toCustomImage(data):
    (c,h,w)=data.shape[1:]
    r=math.ceil(math.sqrt(c))
    img=np.zeros((h*r,w*r))
    for i in range(c):
        y=int(i/r)*h
        x=i%r*w
        img[y:y+h,x:x+w]=data[0,i,:,:]
    #normalize
    mi=np.min(img)
    img=img-mi
    mx = np.max(img)
    img=(img)/mx*255

    return img


def readImage():
    images=[]
    if config.isDir:
        for dirpath, dirnames, filenames in os.walk(config.filePath):
            for file in filenames:
                if file.split('.')[-1] in ['jpg', 'png', 'jpeg', 'bmp']:
                    images.append([cv2.imread(os.path.join(dirpath,file),0),os.path.splitext(file)[0]])
    else:
        images.append([cv2.imread(config.filePath, 0), os.path.splitext(os.path.split(config.filePath)[1])[0]])
    return images


def module_hook(id, fileName,results):
    def print_result(model,inData,output):
        data=output.data.cpu().numpy()
        image=toCustomImage(data)
        item={}

        item['image']=image
        item['layer']=id
        item['name']=fileName
        results.append(item)
    return print_result


def registHook(fileName):
    hook_list = []
    results=[]

    for name, m in net.named_modules():
        if 'relu' in name:
            hook_list.append(m.register_forward_hook(module_hook(name,fileName,results)))
    return hook_list,results


def removeHook(hook_list):
    for hook in hook_list:
        hook.remove()


config=Config()
config.print()

net = CrowdCounter()
net.cuda()
net.eval()

net.load_state_dict(torch.load(config.modelPath))
images=readImage()
for image in images:
    fileName=image[1]
    image=image[0]
    print('Solving',fileName)
    image=image.reshape((1, 1, image.shape[0], image.shape[1]))
    if config.needShowResult or config.needOutput:
        hook_list,results=registHook(fileName)
    density_map=net.forward(image)
    if config.needShowResult or config.needOutput:
        removeHook(hook_list)
        if config.needShowResult:
            dpi = 3
            plt_size = (5 * dpi, 3 * dpi)
            fig=plt.figure(figsize=plt_size)
            fig.canvas.set_window_title(results[0]['name'])
            for i,item in enumerate(results):
                sub = fig.add_subplot(3, 5, i + 1)
                sub.set_title(item['layer'])
                sub.imshow(item['image'])
            plt.show()
        if config.needOutput:
            for item in results:
                mid_path = os.path.join(config.outResultPath,item['layer'])
                os.makedirs(mid_path, exist_ok=True)
                cv2.imwrite('{0}/{1}.png'.format(mid_path, item['name']), item['image'])
    density_map=density_map.data.cpu().numpy()
    sum=np.sum(density_map)
    print('Predict crowd count:',sum)
if config.needOutput:
    print('Results saved at:')
    print(os.path.realpath(config.outResultPath))


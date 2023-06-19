import cv2
import numpy as np
import argparse
import os
from model import NetworkNvidia, myNetworkFromNvidia,LeNet
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms


path = "./test"
y_bias = 186
gray_min = np.array([0, 0, 46])
gray_max = np.array([180, 43, 255])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


img_transform = transforms.Compose([transforms.ToTensor(), 
                                   # transforms.Normalize(*mean_std)
                                   ])
#model = NetworkNvidia() 
model = myNetworkFromNvidia()
model.load_state_dict(torch.load('./weight_my10-epoch.pth'))# my ANN

def test(img):
    model.eval()
    test_loss = 0
    correct = 0
     
    data =Variable(img_transform(img))
    output = model(data)
    print(output)    
    ''' 
    pred = torch.max(output.data, 1)[1]
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    '''
# sigmoid函数
def sigmoid(x):
    return 1. / (1 + np.exp(-x))

# tanh函数
def tanh(x):
    return 2. / (1 + np.exp(-2 * x)) - 1

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
if __name__ == '__main__':

 
    
    net = cv2.dnn.readNetFromONNX('mynvidia.onnx')
     
    filelist = os.listdir(path)
    for file in filelist:
        print(file)
        
        image_name = path + '/'+ file
        
        srcimg = cv2.imread(image_name)
        gray = cv2.cvtColor(srcimg, cv2.COLOR_BGR2GRAY)
        srcimg2 = srcimg[134:200,0:200]
        hsv = cv2.cvtColor(srcimg, cv2.COLOR_BGR2HSV)
        mask_gray = cv2.inRange(hsv, gray_min, gray_max)

        # 二值化
        retval, dst = cv2.threshold(mask_gray, 30, 255, cv2.THRESH_BINARY)
        # 膨胀，白区域变大
        dst = cv2.dilate(dst, None, iterations=2)
        # # 腐蚀，白区域变小
        dst = cv2.erode(dst, None, iterations=6)
        #矩阵切片，把需要的东西提取出来
        hawk = dst[134:200,0:200]
        
        
        #cv2.imshow("hawk.jpg",img2)

        #cv2.waitKey(50)
        #test(hsv)
        #continue 
         
        #srcimg = cv2.normalize(srcimg, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        t1 = cv2.getTickCount()
        blob = cv2.dnn.blobFromImage(hsv, 
                                     #scalefactor=1,
                                     #size= (66, 200),
                                     #mean=[0.485]
                                     )
        net.setInput(blob)
        layer = net.getUnconnectedOutLayersNames()#获取最后一层 
        #layeralls= net.getLayerNames()#获取所有的输出层名称
        #print(layer)
        #前向传播获得信息
        pred = net.forward(layer) 
        
        #print( (pred))
        #print( type(pred))#查看类型 class tuple
        #print( len(pred))#查看元组长度 1
        #print( type(pred[0]))#查看元组0的类型
        
        array_output= pred[0]
       
        #print( array_output)
        
        #print( array_output.ndim) #查看维度
        #print(array_output.shape)#输出行数和列数 1x10 与netron 查看结果一样
        #print(array_output.size)#输出总共有多少元素  上边相乘的结果 
      
        
        t3 = cv2.getTickCount()
        sec = (t3 - t1)
        label = 'Inference time: %.2f ms' % (sec * 1000.0 /  cv2.getTickFrequency())
        #print(label)
        
        feature_map = array_output[0]
        print("result:"+ str(feature_map))
        #print("softmax:"+ str(softmax(feature_map)))
        
 

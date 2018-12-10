import os
import sys
import cv2
from model import EDSR
from model import MDSR
from model import NewNet
from torchvision import transforms
import numpy as np
import torch

img_path = 'test/head.jpg'
img = cv2.imread( img_path )
#img = cv2.resize( img, (int(img.shape[1]/2), int( img.shape[0]/2) ), cv2.INTER_CUBIC )
print ('before', img.shape )
np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
input = torch.from_numpy(np_transpose).float()
input = input.mul_(255 / 255)
input = input.cuda()
input =  torch.unsqueeze(input ,0 )

scale_list = [1]
#net = NewNet.NewNet(scale_list)
#net = MDSR.MDSR(scale_list)
net = EDSR.EDSR(scale_list)
net = net.cuda()
net.eval()

res = net.forward_pred( input,1)

res = res.cpu()
res = res.view( res.size()[1], res.size()[2], res.size()[3] )
res_np = res.data.numpy()
res_img = res_np.transpose((1, 2, 0))
res_img = np.array( res_img, dtype = np.uint8)
print (res_img.shape)
cv2.imwrite( 'test/head_1.png', res_img )

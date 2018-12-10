import os
import sys
import cv2
import torch
import argparse
import numpy as np
from comm import comm

if __name__ == "__main__" :
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', required = True, help = 'edsr mdsr newnet')
    ap.add_argument('-S', required = False, help = 'scale', default = 1)
    ap.add_argument('-s', required = True, help = 'scale list')
    ap.add_argument('-g', required = False, help = 'gpu id', default = 0)
    ap.add_argument('-p', required = True, help = 'model path')
    ap.add_argument('-i', required = True, help = 'img path')
    ap.add_argument('-o', required = True, help = 'output path')

    args = vars(ap.parse_args())

    model_type = args['m']
    scale_list = [ int(scale) for scale in args['s'].split(',')]
    scale = int( args['S'] )
    gpu_id = int( args['g'] )
    model_path = args['p']
    img_path = args['i']
    output_path = args['o']

    torch.cuda.set_device(gpu_id)
    img = cv2.imread( img_path )

    print ('before size', img.shape )
    np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
    input = torch.from_numpy(np_transpose).float()
    input = input.mul_(255 / 255)
    input = input.cuda()
    input =  torch.unsqueeze(input ,0 ) 


    scale_list = [scale]
    net = comm.get_model(model_type, scale_list, model_path)
    net = net.cuda()
    net.eval()

    res = net.forward_pred( input, scale)

    res = res.cpu()
    res = res.view( res.size()[1], res.size()[2], res.size()[3] )
    res_np = res.data.numpy()
    res_img = res_np.transpose((1, 2, 0))
    res_img = np.array( res_img, dtype = np.uint8)
    print ('after size:', res_img.shape)
    cv2.imwrite( output_path, res_img )



import os
import sys
import cv2
import torch
import random
import numpy as np
import torch.utils.data as data

class MyDataSet(data.Dataset):
    def __init__( self, data_dir, scale_list, mode = 'train', batch_size = 16, input_img_size = 48, repeat = 1 ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.input_img_size = input_img_size
        self.scale_list = scale_list
        self.scale_index = 0
        self.mode = mode
        self.repeat = repeat

        random.seed( random.randint(0,10000000 ) )
        self.batch_id = 0
        self.load_data()

    def load_data( self ):
        self.data_map = {}
        scale_list = self.scale_list.copy()
        scale_list.append(1)
        scale_list = list(set(scale_list) )
        for scale in scale_list:
            if scale not in self.data_map.keys():
                self.data_map[scale] = []
            self.data_map[scale] = self.get_all_img( os.path.join( self.data_dir, 'X' + str( scale ) ) )
             

    def get_all_img( self, path ):
        def is_image_file(filename):
            return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

        img_list = []
        for root,dirs,filenames in os.walk( path ):
            for filename in filenames:
                imgname = os.path.join( root, filename )
                if is_image_file( imgname ) == True:
                    img_list.append( imgname )
        img_list.sort()
        return img_list

    def np2Tensor( self, data, rgb_range = 255 ):
        img = self.jpeg_compression(data)
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)
        return tensor

    def jpeg_compression(self, img ):
        if self.mode == 'train':
            quality = random.randint( 50, 90 )
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            result, encimg = cv2.imencode('.jpg', img, encode_param)
            decimg = cv2.imdecode(encimg, 1)
            return decimg 
        else :
            return img;

    def getdata( self, input_f, label_f ):
        input_img = cv2.imread( input_f )
        label_img = cv2.imread( label_f )

        if self.mode == 'train':
            ix = random.randrange( 0, int(input_img.shape[0] - self.input_img_size ) )
            iy = random.randrange( 0, int(input_img.shape[1] - self.input_img_size ) )
            input_img = input_img[ix:ix+self.input_img_size, iy:iy+self.input_img_size, : ]
            input_img = self.np2Tensor(input_img )

            ix *= self.scale
            iy *= self.scale
            label_img = label_img[ix:ix+self.input_img_size * self.scale, iy:iy+self.input_img_size * self.scale, : ]
            label_img = self.np2Tensor( label_img )
        else :
            ix,iy = input_img.shape[0],input_img.shape[1]
            input_img = input_img[:ix, :iy]
            input_img = self.np2Tensor(input_img )

            ix *= self.scale
            iy *= self.scale
            label_img = label_img[:ix, :iy ]
            label_img = self.np2Tensor( label_img )

        return input_img, label_img

    def changestatus( self ):
        if self.mode == 'train':
            if( self.batch_id == 0 ):
                self.scale = random.choice( self.scale_list )
            self.batch_id = ( self.batch_id + 1 ) % self.batch_size
        else:
             if( self.batch_id == 0 ):
                 self.scale = self.scale_list[self.scale_index]
                 self.scale_index = (self.scale_index + 1 ) % len(self.scale_list )
             self.batch_id = ( self.batch_id + 1 ) % len( self.data_map[1] )

    def __getitem__(self, index ):
        self.changestatus()
        index = index % len(self.data_map[1])

        label_f, input_f = self.data_map[1][index], self.data_map[self.scale][index]
        input_img, label_img = self.getdata( input_f, label_f )

        return self.scale, input_img, label_img

    def __len__(self):
        if self.mode == 'train':
            return len(self.data_map[1]) * self.repeat
        else:
            return len(self.data_map[1]) * len(self.scale_list)


#data_dir = '/data/user/data1/lockeliu/learn/data/super_solution/process_data/DIV2K/train/'
#input_img_size = 48
#scale_list = [2,3,4]
#repeat = 100
#
#a = MyDataSet( data_dir, input_img_size, scale_list, repeat )
#print (a.__len__() )
#input_img, label_img = a.__getitem__( 101 )
#print( input_img.size(), label_img.size() )

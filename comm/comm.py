import os
from model import EDSR
from model import MDSR
from model import NewNet

def get_model(model_type, scale_list, model_path):
    model_type = model_type.lower()
    if model_type == 'edsr':
        return EDSR.EDSR(scale_list, model_path)
    elif model_type == 'mdsr':
        return MDSR.MDSR( scale_list, model_path )
    elif model_type == 'newnet':
        return NewNet.NewNet( scale_list, model_path )
    else:
        print("no this model_type " + model_type)
        exit(-1)

def checkandmkdir(path):
    dir_path = '/'.join(path.split('/')[:-1])
    if( os.path.exists(dir_path) == False):
        os.makedirs(dir_path)

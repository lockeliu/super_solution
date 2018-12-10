import os
import sys
import argparse
from solver import solver,solver_gan

if __name__ == "__main__" :
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', required = True, help = 'edsr mdsr srgan')
    ap.add_argument('-t', required = True, help = 'train data dir')
    ap.add_argument('-v', required = True, help = 'val data dir')
    ap.add_argument('-s', required = False, help = 'scale list', default = 2)
    ap.add_argument('-b', required = False, help = 'batch size', default = 64)
    ap.add_argument('-i', required = False, help = 'input size', default = 48)
    ap.add_argument('-r', required = False, help = 'repeat', default = 10)
    ap.add_argument('-e', required = False, help = 'epoch', default = 100)
    ap.add_argument('-l', required = False, help = 'lr', default = 0.001)
    ap.add_argument('-g', required = False, help = 'gpu num', default = 4)
    ap.add_argument('-p', required = True, help = 'model_path')
    ap.add_argument('-dp', required = True, help = 'D_model_path')
    ap.add_argument('-is_gan', required = False, help = '0:no gan 1 :gan', default = 0)

    args = vars(ap.parse_args())

    model_type = args['m']
    train_data_dir = args['t']
    val_data_dir = args['v']
    scale_list = [ int(scale) for scale in args['s'].split(',')]
    batch_size = int( args['b'] )
    input_img_size = int( args['i'] )
    repeat = int( args['r'] )
    epoch = int( args['e'] )
    lr = float( args['l'] )
    gpu_num = int( args['g'] )
    model_path = args['p']
    d_model_path = args['dp']
    is_gan = int(args['is_gan'] )

    if is_gan == 1:
        trainer = solver_gan.Trainer( model_type, scale_list, train_data_dir, val_data_dir, model_path, d_model_path, batch_size, input_img_size, repeat, epoch, lr, gpu_num )
    else:
        trainer = solver.Trainer(model_type, scale_list, train_data_dir, val_data_dir, model_path, batch_size, input_img_size, repeat, epoch, lr, gpu_num )
    trainer.run();

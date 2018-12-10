import os
import sys
import cv2
import math
import argparse
import multiprocessing

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def checkandmkdir( path ):
	if os.path.exists( path ) == False:
		os.makedirs( path )

def gendata( imgname , output_data_dir, filename ):
    img = cv2.imread( imgname )
    scale_list = [1,2,3,4]

    for scale in scale_list:
        out_dir = os.path.join( output_data_dir, 'X' + str(scale) )
        checkandmkdir( out_dir )
        out_filename = os.path.join( out_dir, filename )
        if scale != 1:
            new_img = cv2.resize( img, ( math.floor(img.shape[1] / scale) , math.floor(img.shape[0] / scale) ),  interpolation=cv2.INTER_CUBIC )
        else :
            new_img = img
        cv2.imwrite( out_filename, new_img )
			

if __name__ == "__main__" :
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', required = True, help = 'data_dir')
    ap.add_argument('-o', required = True, help = 'output_data_dir')
    ap.add_argument('-n', required = True, help = 'num_worker')

    args = vars(ap.parse_args())

    data_dir = args['d']
    output_data_dir = args['o']
    num_worker = int(args['n'])

    if data_dir[-1] != '/':
        data_dir += '/'

    if output_data_dir[-1] != '/':
        output_data_dir += '/'

    pool = multiprocessing.Pool(processes = num_worker )

    for root,dirs,filenames in os.walk( data_dir ):
        for filename in filenames:
            if is_image_file( filename ) == True:
                imgname = os.path.join( root, filename )
                output_dir = root.replace( data_dir, output_data_dir )
                pool.apply_async( gendata, ( imgname, output_dir, filename ) )

    pool.close()
    pool.join()

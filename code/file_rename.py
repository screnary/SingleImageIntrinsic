# rename image files based on the index and test split files
# for video temporal consistency evaluation
import os
import sys
from shutil import copy
from os.path import join, dirname, abspath
import argparse


args = argparse.ArgumentParser(description='rename pred files according to split file')
args.add_argument('--proj_root', type=str,
                  default='/home/wzj/intrinsic/intrinsic_image_project',
                  help='the pwd of intrinsic image project.py')
args.add_argument('--input_root', type=str,
                  default='framewise-ckpoints-v8_self-sup+ms+fd+pers-MPI-main-RD-sceneSplit-w_ssim-0.5-w_grad-1.5/log',
                  help='the pwd of intrinsic image project.py')
args.add_argument('--res_folder', type=str,
                  default='test-imgs_ep200',
                  help='the result folder')
args.add_argument('--out_folder', type=str, 
                  default='test-imgs_ep200_renamed',
                  help='the output folder')
opts = args.parse_args()

if __name__ == '__main__':

    fname = 'MPI_main_sceneSplit-fullsize-NoDefect-test-video.txt'
    src_root = join(opts.proj_root, opts.input_root, opts.res_folder)
    tar_root = join(opts.proj_root, opts.input_root, opts.out_folder)

    if not os.path.exists(tar_root):
        os.mkdir(tar_root)

    with open(join(opts.proj_root, 'datasets', 'MPI', fname), 'r') as fid:
        lines = fid.readlines()
        for i in range(len(lines)):
            img_name = lines[i].strip().split('.')[0]

            src_pred_r = '{}_reflect-pred.png'.format(i)
            src_pred_s = '{}_shading-pred.png'.format(i)
            src_real_r = '{}_reflect-real.png'.format(i)
            src_real_s = '{}_shading-real.png'.format(i)
            
            tar_pred_r = '{}_reflect-pred.png'.format(img_name)
            tar_pred_s = '{}_shading-pred.png'.format(img_name)
            tar_real_r = '{}_reflect-real.png'.format(img_name)
            tar_real_s = '{}_shading-real.png'.format(img_name)

            copy(join(src_root, src_pred_r), join(tar_root, tar_pred_r))
            copy(join(src_root, src_pred_s), join(tar_root, tar_pred_s))
            copy(join(src_root, src_real_r), join(tar_root, tar_real_r))
            copy(join(src_root, src_real_s), join(tar_root, tar_real_s))

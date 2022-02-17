# echo "for DI results"
# python file_rename.py --input_root framewise-ckpoints-direct_intrinsics-MPI-main-RD-sceneSplit/log --res_folder test-imgs_ep225 --out_folder test-imgs_ep225_renamed

# echo "for revisiting results"
# python file_rename.py --input_root IntrinsicImage-master/results/test --res_folder RD_MPI-main-clean-video --out_folder RD_MPI-main-clean-video-renamed

# echo "for video results rename-1"
# python file_rename.py --input_root /home/wzj/intrinsic/code/video-ckpoints-v8_self-sup+ms+fd+pers-MPI-main-RD-sceneSplit-w_flow-50.0-lambda_r-4.0-lambda_s-0.0/log --res_folder test-imgs_ep250 --out_folder test-imgs_ep250-renamed

# echo "for video results rename-2"
# python file_rename.py --input_root /home/wzj/intrinsic/code/video-ckpoints-v8_self-sup+ms+fd+pers-MPI-main-RD-sceneSplit-w_flow-50.0-lambda_r-1.0-lambda_s-1.0/log --res_folder test-imgs_ep250 --out_folder test-imgs_ep250-renamed


echo "for video results rename-3 fromV8"
python file_rename.py --input_root /home/wzj/intrinsic/code/video-ckpoints-fromV8-v8_self-sup+ms+fd+pers-MPI-main-RD-sceneSplit-w_flow-50.0-lambda_r-10.0-lambda_s-5.0/log --res_folder test-imgs_ep5 --out_folder test-imgs_ep5-renamed

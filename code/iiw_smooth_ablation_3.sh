echo "varying w_rs from 0 to 4, keeping w_ss to be equal to w_rs"
#python main_iiw_oneway_pixsup_v7.py --w_rs 0.5 --w_ss 0.5 --gpu_id 1
#python main_iiw_oneway_pixsup_v7.py --w_rs 1.0 --w_ss 1.0 --gpu_id 1
python main_iiw_oneway_pixsup_v7.py --w_rs 2.0 --w_ss 2.0 --gpu_id 0 --continue_train --which_epoch 4
python main_iiw_oneway_pixsup_v7.py --w_rs 4.0 --w_ss 4.0 --gpu_id 0
echo "finished setting 3"


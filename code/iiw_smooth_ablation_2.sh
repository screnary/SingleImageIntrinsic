echo "varying w_ss from 0 to 4, keeping w_rs to be 0"
#python main_iiw_oneway_pixsup_v7.py --w_rs 0.0 --w_ss 0.5
#python main_iiw_oneway_pixsup_v7.py --w_rs 0.0 --w_ss 1.0
python main_iiw_oneway_pixsup_v7.py --w_rs 0.0 --w_ss 2.0 --gpu_id 0
python main_iiw_oneway_pixsup_v7.py --w_rs 0.0 --w_ss 4.0 --gpu_id 0
echo "finished setting 2"


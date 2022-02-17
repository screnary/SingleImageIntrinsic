echo "varying w_rs from 0 to 4, keeping w_ss to be 0"
python main_iiw_oneway_pixsup_v7.py --w_rs 0.0 --w_ss 0.0
python main_iiw_oneway_pixsup_v7.py --w_rs 0.5 --w_ss 0.0
python main_iiw_oneway_pixsup_v7.py --w_rs 1.0 --w_ss 0.0
python main_iiw_oneway_pixsup_v7.py --w_rs 2.0 --w_ss 0.0
python main_iiw_oneway_pixsup_v7.py --w_rs 4.0 --w_ss 0.0
echo "finished setting 1"

echo "varying w_ss from 0 to 4, keeping w_rs to be 0"
python main_iiw_oneway_pixsup_v7.py --w_ss 0.5 --w_rs 0.0
python main_iiw_oneway_pixsup_v7.py --w_ss 1.0 --w_rs 0.0
python main_iiw_oneway_pixsup_v7.py --w_ss 2.0 --w_rs 0.0
python main_iiw_oneway_pixsup_v7.py --w_ss 4.0 --w_rs 0.0
echo "finished setting 2"

#echo "varying w_rs from 0 to 4, keeping w_ss to be equal with w_rs"
#python main_iiw_oneway_pixsup_v7.py --w_rs 0.5 --w_ss 0.5
#python main_iiw_oneway_pixsup_v7.py --w_rs 1.0 --w_ss 1.0
#python main_iiw_oneway_pixsup_v7.py --w_rs 2.0 --w_ss 2.0
#python main_iiw_oneway_pixsup_v7.py --w_rs 4.0 --w_ss 4.0
#echo "finished setting 3"



echo "evaluation of ablation studies"
echo "rs=0, ss=0"
python main_iiw_oneway_pixsup_v7.py --w_rs 0.0 --w_ss 0.0 --gpu_id 0 --phase test --best_ep 10
echo "rs=0, ss=0.5"
python main_iiw_oneway_pixsup_v7.py --w_rs 0.0 --w_ss 0.5 --gpu_id 0 --phase test --best_ep 6
echo "rs=0, ss=1.0"
python main_iiw_oneway_pixsup_v7.py --w_rs 0.0 --w_ss 1.0 --gpu_id 0 --phase test --best_ep 9 
echo "rs=0, ss=2.0"
python main_iiw_oneway_pixsup_v7.py --w_rs 0.0 --w_ss 2.0 --gpu_id 0 --phase test --best_ep 7 
echo "rs=0, ss=4.0"
python main_iiw_oneway_pixsup_v7.py --w_rs 0.0 --w_ss 4.0 --gpu_id 0 --phase test --best_ep 8 
echo "rs=0.5, ss=0"
python main_iiw_oneway_pixsup_v7.py --w_rs 0.5 --w_ss 0.0 --gpu_id 0 --phase test --best_ep 9 
echo "rs=1.0, ss=0"
python main_iiw_oneway_pixsup_v7.py --w_rs 1.0 --w_ss 0.0 --gpu_id 0 --phase test --best_ep 8 
echo "rs=2.0, ss=0"
python main_iiw_oneway_pixsup_v7.py --w_rs 2.0 --w_ss 0.0 --gpu_id 0 --phase test --best_ep 8 
echo "rs=4.0, ss=0"
python main_iiw_oneway_pixsup_v7.py --w_rs 4.0 --w_ss 0.0 --gpu_id 0 --phase test --best_ep 7 
echo "rs=0.5, ss=0.5"
python main_iiw_oneway_pixsup_v7.py --w_rs 0.5 --w_ss 0.5 --gpu_id 0 --phase test --best_ep 10 
echo "rs=1., ss=1."
python main_iiw_oneway_pixsup_v7.py --w_rs 1.0 --w_ss 1.0 --gpu_id 0 --phase test --best_ep 10 
echo "rs=2.0, ss=2.0"
python main_iiw_oneway_pixsup_v7.py --w_rs 2.0 --w_ss 2.0 --gpu_id 0 --phase test --best_ep 10 
echo "rs=4.0, ss=4.0"
python main_iiw_oneway_pixsup_v7.py --w_rs 4.0 --w_ss 4.0 --gpu_id 0 --phase test --best_ep 8 
echo "finished"
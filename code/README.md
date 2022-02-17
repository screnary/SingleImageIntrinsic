2021.03.06
For Essay Writing

---
1. check out which setting used for the final results

  MPI_RD dataset:
  ours:
- plain: v8_3 (ms,multiscale)

- w/o FDV: v8_2 (ms+pers, multiscale+perspective)

- w/o FDC: v8_1 (ms+fd, multiscale+feature divergence)

- Ours: v8 (ms+fd+pers)

  

  ```shell
  # 43994
  conda activate pytorch
  python main_mpi_RD_v8_3.py --w_ssim 0.0 --w_grad 0.0
  
  python main_mpi_RD_v8_3.py --w_ssim 1.0 --w_grad 0.0
  
  python main_mpi_RD_v8_3.py --w_ssim 0.0 --w_grad 1.0
  ```

  

  

  IIW dataset:
  ours:

  oneway_v7-ep12

- reflectance local smooth:
0, 0.5, 1.0, 2.0, 4.0
- shading dense smooth:
0, 0.5, 1.0, 2.0, 4.0

settings:
(r, s)
(0, (0~4))
((0~4), 0)
((0~4), ${r})

```shell
# setting 1
python main_iiw_oneway_pixsup_v7.py --w_rs 0.0 --w_ss 0.0
python main_iiw_oneway_pixsup_v7.py --w_rs 0.5 --w_ss 0.0
python main_iiw_oneway_pixsup_v7.py --w_rs 1.0 --w_ss 0.0
python main_iiw_oneway_pixsup_v7.py --w_rs 2.0 --w_ss 0.0
python main_iiw_oneway_pixsup_v7.py --w_rs 4.0 --w_ss 0.0

# setting 2
python main_iiw_oneway_pixsup_v7.py --w_rs 0.0 --w_ss 0.5
python main_iiw_oneway_pixsup_v7.py --w_rs 0.0 --w_ss 1.0
python main_iiw_oneway_pixsup_v7.py --w_rs 0.0 --w_ss 2.0
python main_iiw_oneway_pixsup_v7.py --w_rs 0.0 --w_ss 4.0

# setting 3
python main_iiw_oneway_pixsup_v7.py --w_rs 0.5 --w_ss 0.5
python main_iiw_oneway_pixsup_v7.py --w_rs 1.0 --w_ss 1.0
python main_iiw_oneway_pixsup_v7.py --w_rs 2.0 --w_ss 2.0
python main_iiw_oneway_pixsup_v7.py --w_rs 4.0 --w_ss 4.0

```
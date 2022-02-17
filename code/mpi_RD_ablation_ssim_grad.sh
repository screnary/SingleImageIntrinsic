echo ablation study evaluation phase of MPI_RD ssim and grad weights
echo ssim 0.0, grad 0.0
python main_mpi_RD_v8_3.py --w_ssim 0.0 --w_grad 0.0 --phase test --best_ep 150
echo ssim 0.0, grad 1.0
python main_mpi_RD_v8_3.py --w_ssim 0.0 --w_grad 1.0 --phase test --best_ep 150
echo ssim 1.0, grad 0.0
python main_mpi_RD_v8_3.py --w_ssim 1.0 --w_grad 0.0 --phase test --best_ep 150
echo finished

inhirent from IntrinsicImage Project. 'Revisiting Deep Intrinsic Image Decompositions' (CVPR 2018)


steps:
1. propagate_invalid_masks.m   :: using optical flow to propagate invalid masks in the same scene, need to run before MPI_data_prepare.m, to get the LLE inpainting masks.

2. MPI_data_prepare_step1.m  :: main processing. (reconstruction and inpainting)

3. MPI_data_prepare_step2.m  :: 将shading的颜色抹除，只用gray的luminance channel。完成生成数据。

4. MPI_data_prepare_step3.m  :: 检查有无无效区域，不满足I=A×S的。

5. MPI_data_prepare_step4.m  :: 检查refined shading与original shading的颜色分布是否一致，使用颜色统计直方图的变化趋势一致性约束，消除前后帧亮度抖动。

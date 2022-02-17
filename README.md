Single Image Intrinsic Decomposition with Discriminative Feature Encoding

## 工作内容介绍：

《基于可区分性特征编码的场景本征图像分解》

发表于 ICCVW-2019, Single Image Intrinsic Decomposition with Discriminative Feature Encoding.

本征图像分解是研究从图像中恢复物理场景属性的计算机视觉任务。输入一张图片，经过本征分解后，可以得到图片种所描述的三维场景的不同方面的本征图像（如描述纹理的Albedo图像，与描述光照与几何的Shading图像）。然而，从一张图像中恢复出多张本征图像，本质上是一个病态的数学问题。近年来深度学习在图像识别与解析等领域的成功，吸引了大量学者用深度网络解决本征图像分解问题。在本工作中，我们基于图像特征提取过程中，不同本征成分所需重构元素应具有一定可分性的假设，提出了一个新的双路本征分解网络。为了利用不同本征成分在特征空间的可分性，本方法使用两个特征提取器分别从同一张输入图像中提取特征，这些特征在特征分离（Feature Divergence Loss）的作用下，在特征空间相互分散。然而，仅仅增强不同本征成分在特征空间的可分性，并不能保证同一本征成分具有一致且合理的分布。因此，本方法通过将编码器用作感知特征一致性约束的特征提取器，来约束特征编码符合真实分布（Feature Distribution Constraint）。除此之外，本工作还针对现有数据集的光照不一致问题进行了优化，得到了更加适用于本征分解任务的训练、测试数据，并在实验室主页公开了相关数据与代码。本方法的分解准确率在常用本征图像分解数据集上达到或超越了现有state-of-the-art方法。

## 📁文件格式整理

```shell
<intrinsic_project>
├───intrinsic_image_project               # 单张图像本征分解，工程代码目录
│ └───code
│     	│  README.md                      # 详细的命令记录
│     	│  main_iiw_oneway_pixsup_v7.py   # iiw 数据集的训练、测试代码主文件
│     	│  main_mit.py										# MIT 数据集的训练、测试代码主文件
│     	│  main_mit_v1.py									# 在MPI上预训练，然后在MIT上微调的主文件
│     	│  main_mpi_new.py 								# 在MPI数据集上的训练、测试代码主文件
│     	│  main_mpi_RD_*.py								# 在优化后的数据集MPI_RD上的训练、测试代码主文件
│     	│  my_data_*.py										# 数据载入函数文件
│     	│  networks_*.py									# 网络结构代码
│     	│  trainer_*.py  									# 训练策略脚本
│     	│  file_raname.py                 # 输出结果重命名脚本，为了与gt比较，将index对应为文件名
│     	│  iiw_smooth_ablation_*.sh       # iiw数据集上，smooth权重大小消融实验命令脚本
│     	│  mpi_RD_*.sh                    # MPI_RD数据集上，实验命令脚本
│     	│  run_file_raname.sh             # 文件重命名命令脚本
│     	│
│     	└───data                          # data loader 代码目录
│     	└───pytorch_ssim                  # ssim 指标计算代码目录
│     	└───utils                         # 工具函数代码目录
│     	└───configs                       # 网络训练设置文件目录
│     	└───other_versions                # 实验过程中的其他版本代码目录
│     	└───features                      # 不同网络结构的特征嵌入结果目录
│ │
│ └───comparison_method                   # 对比方法
│ │
│ └───datasets                            # 数据集，该目录下的数据转移到更外层data目录存储
│     	└───IIW                           # IIW数据集
│     	└───MPI                           # MPI数据集
```

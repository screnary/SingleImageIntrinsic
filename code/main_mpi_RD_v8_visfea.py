import torch
import torchvision
from torch.utils.data import DataLoader
import my_data_RD as my_data
#from configs.intrinsic_mpi_RD_v8 import opt
from configs.intrinsic_mpi_RD_v8_visfea import opt
import trainer_mpi_RD_v8_visfea as Trainer
from utils import save_eval_images, check_dir, visualize_inspector
import numpy as np
import pdb

"""
v3: use vgg19; use L1 loss rather than MSE loss, discard gradient and ssim,
    enlarge idt loss weight, change fea_div_dict
v4: use gradient loss, average across 3 channels (rgb), add loss weight
v6: set grad threshold for gt_grad, only large grad values are considered
v7: use mask for crop window selection when feed data
v8: perceptual loss use L2 + cosine, more strict trian patch selection
v9: Grad use 3 channels and x y splited, preserve_info_loss, patch select 0.35,
    fea_distance_loss: cos 0.85, L2 0.15, lr use wave pattern
v10: cropsize=256, L1 loss weight = 30.0
v11: cropsize=288, L1 loss weight = 25.0, lambda_r = lambda_b = 1.0, use
    div_dict for perceptual, use cosine loss for preserve_info
RD: refined data, gray scale shading, I=A.*S; modified from v11
    ms+fd+pers
RD_v8: align_corners=False, conv3 has pad 1, feat_dict=[low,mid,mid2,deep,out], dr = [1,1,1]
"""
check_dir(opt.output_root)
check_dir(opt.logger.log_dir)


# loss_inspector = {'loss_total': [], 'loss_idt': [], 'loss_ssim': [], 'loss_grad': [], 'iter_step': []}

def extract_fea(trainer, dataset, epoch, loss_inspector, save_pred_results=False):
    opt.is_train = False
    # Todo:batch_size should be 1, or change save_images function
    batch_size = 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch_num = len(dataset) // batch_size
    fea_list_r = []
    fea_list_s = []

    for batch_idx, samples in enumerate(dataloader):
        if batch_idx >= 300:
            break
        trainer.set_input(samples)
        fea_s, fea_r = trainer.forward_feas()
        fea_list_s.append(fea_s)
        fea_list_r.append(fea_r)
        # pdb.set_trace()

    return np.concatenate(fea_list_s, axis=0), np.concatenate(fea_list_r, axis=0)


def basic_settings(opt):
    run_settings = {}
    """setup datasets"""
    if 'MPI' in opt.data.name:
        train_dataset = my_data.DatasetIdMPI_RD(opt.data, is_train=True, cropped=True)  # DatasetIDMPI_mask
        test_dataset = my_data.DatasetIdMPI_RD(opt.data, is_train=False, cropped=True)
    elif 'MIT' in opt.data.name:
        train_dataset = my_data.DatasetIdMIT(opt.data, is_train=True)
        test_dataset = my_data.DatasetIdMIT(opt.data, is_train=False)
    else:
        raise NotImplementedError

    """setup model trainer"""
    if opt.train.trainer_mode == 'GAN':
        model_trainer = Trainer.Trainer_GAN(opt)
    else:  # Basic
        model_trainer = Trainer.Trainer_Basic(opt)

    # evaluate(trainer, test_dataset, 0)
    # pdb.set_trace()
    """training process"""
    train_loss_inspector = {'loss_total': [], 'loss_idt': [], 'loss_ssim': [], 'loss_grad': [], 'loss_fea_diver': [],
                            'loss_preserve_info': [], 'lr': [],
                            'loss_perspective': [], 'loss_fea_extract': [], 'step': []}
    test_loss_inspector = {'loss_total': [], 'loss_idt': [], 'loss_ssim': [], 'loss_grad': [], 'loss_fea_diver': [],
                           'loss_preserve_info': [], 'lr': [],
                           'loss_perspective': [], 'loss_fea_extract': [], 'step': []}
    if opt.train.trainer_mode == 'GAN':
        train_loss_inspector['loss_GAN_G'] = []
        train_loss_inspector['loss_GAN_D_s'] = []
        train_loss_inspector['loss_GAN_D_r'] = []

        test_loss_inspector['loss_GAN_G'] = []
        test_loss_inspector['loss_GAN_D_s'] = []
        test_loss_inspector['loss_GAN_D_r'] = []

    run_settings['train_dataset'] = train_dataset
    run_settings['test_dataset'] = test_dataset
    run_settings['model_trainer'] = model_trainer
    run_settings['train_loss_inspector'] = train_loss_inspector
    run_settings['test_loss_inspector'] = test_loss_inspector

    return run_settings


def test_ms(epoch=None):
    from configs.intrinsic_mpi_RD_v8_3_visfea import opt as opt1
    settings = basic_settings(opt1)
    test_dataset = settings['test_dataset']
    model_trainer = settings['model_trainer']
    test_loss_inspector = settings['test_loss_inspector']

    model_trainer.resume(model_trainer.gen_split, 'G_decompose', epoch_name=epoch)
    if opt.train.trainer_mode == 'GAN':
        model_trainer.resume(model_trainer.dis_S, 'D_S', epoch_name=epoch)
        model_trainer.resume(model_trainer.dis_R, 'D_R', epoch_name=epoch)

    feature_s, feature_r = extract_fea(model_trainer, test_dataset, epoch=epoch,
                            loss_inspector=test_loss_inspector, save_pred_results=True)
    # pdb.set_trace()
    np.save(opt1.logger.log_dir + '/feature_s.npy', feature_s)
    np.save(opt1.logger.log_dir + '/feature_r.npy', feature_r)


def test_ms_fd_pers(epoch=None):
    from configs.intrinsic_mpi_RD_v8_visfea import opt as opt2
    settings = basic_settings(opt2)
    test_dataset = settings['test_dataset']
    model_trainer = settings['model_trainer']
    test_loss_inspector = settings['test_loss_inspector']

    model_trainer.resume(model_trainer.gen_split, 'G_decompose', epoch_name=epoch)
    if opt.train.trainer_mode == 'GAN':
        model_trainer.resume(model_trainer.dis_S, 'D_S', epoch_name=epoch)
        model_trainer.resume(model_trainer.dis_R, 'D_R', epoch_name=epoch)

    feature_s, feature_r = extract_fea(model_trainer, test_dataset, epoch=epoch,
                                       loss_inspector=test_loss_inspector, save_pred_results=True)
    np.save(opt2.logger.log_dir + '/feature_s.npy', feature_s)
    np.save(opt2.logger.log_dir + '/feature_r.npy', feature_r)


if __name__ == '__main__':
    #train()
    test_ms(epoch=80)
    test_ms_fd_pers(epoch=200)
    #test(epoch=75)
    #test(epoch=150)

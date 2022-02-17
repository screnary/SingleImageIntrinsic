import torch
import torchvision
from torch.utils.data import DataLoader
import my_data as my_data
from configs.intrinsic_mpi_RD_v10 import opt
import trainer_mpi_RD_v10_test as Trainer
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
RD_v9: share low level encoders
RD_v10_test: pad images to eliminate corner artifacts
"""
check_dir(opt.output_root)
check_dir(opt.logger.log_dir)


# loss_inspector = {'loss_total': [], 'loss_idt': [], 'loss_ssim': [], 'loss_grad': [], 'iter_step': []}
def train_one_epoch(trainer, dataset, epoch, loss_inspector):
    opt.is_train = True
    dataloader = DataLoader(dataset, batch_size=opt.data.batch_size, shuffle=True)
    batch_num = len(dataset)//opt.data.batch_size

    if opt.train.trainer_mode == 'GAN':
        if epoch % 5 < 4:
            trainer.set_gan_gd_loop(G_loop=3, D_loop=1)
        else:
            trainer.set_gan_gd_loop(G_loop=1, D_loop=3)

    log_list = []

    for batch_idx, samples in enumerate(dataloader):
        trainer.set_input(samples)
        trainer.optimize_parameters()

        losses = trainer.get_current_errors()

        if np.isnan(losses['loss_total'].cpu().item()):
            print('Warning: nan loss!')
            pdb.set_trace()
        loss_inspector['loss_total'].append(losses['loss_total'].cpu().item())
        loss_inspector['loss_idt'].append(losses['idt_S'].cpu().item() + losses['idt_R'].cpu().item())
        loss_inspector['loss_ssim'].append(losses['ssim_S'].cpu().item() + losses['ssim_R'].cpu().item())
        loss_inspector['loss_grad'].append(losses['grad_S'].cpu().item() + losses['grad_R'].cpu().item())
        loss_inspector['loss_fea_diver'].append(losses['fea_divergence'].cpu().item())
        loss_inspector['loss_perspective'].append(losses['perspective'].cpu().item())
        loss_inspector['loss_fea_extract'].append(losses['fea_extract'].cpu().item())
        loss_inspector['loss_preserve_info'].append(losses['preserve_info'].cpu().item())
        loss_inspector['lr'].append(trainer.get_lr())

        if opt.train.trainer_mode == 'GAN':
            loss_inspector['loss_GAN_G'].append(losses['gan_G'].cpu().item())
            loss_inspector['loss_GAN_D_s'].append(losses['gan_D_s'].cpu().item())
            loss_inspector['loss_GAN_D_r'].append(losses['gan_D_r'].cpu().item())

        loss_inspector['step'].append((epoch-1) * batch_num + batch_idx + 1)

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, batch_num,
                100.0*batch_idx/batch_num, losses['loss_total'].cpu().item()))
            log_detail = trainer.loss_log(losses)
            print(log_detail)
            log_list.append(log_detail)
        if opt.train.save_train_img and (epoch % opt.train.save_per_n_ep_train) == 0:
            visuals = trainer.get_current_visuals()
            img_dir = opt.logger.log_dir + 'train-imgs_ep' + str(epoch)  # '../checkpoints/log/'
            check_dir(img_dir)
            save_eval_images(visuals, img_dir, batch_idx, opt)
    if epoch % opt.train.save_per_n_ep == 0:
        with open(opt.logger.log_dir + '/train_loss_log-ep' + str(epoch) + '.txt', 'w') as f:
            f.writelines(["%s\n" % item for item in log_list])


def evaluate_one_epoch(trainer, dataset, epoch, loss_inspector, save_pred_results=False):
    opt.is_train = False
    # Todo:batch_size should be 1, or change save_images function
    batch_size = opt.data.batch_size_test if not save_pred_results else 1
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch_num = len(dataset) // batch_size

    log_list = []
    loss_total = []
    loss_idt = []
    loss_ssim = []
    loss_grad = []
    loss_fea_diver = []
    loss_perspective = []
    loss_fea_extract = []
    loss_preserve_info = []
    loss_GAN_G = []
    loss_GAN_D_s = []
    loss_GAN_D_r = []

    for batch_idx, samples in enumerate(dataloader):
        trainer.set_input(samples)
        trainer.inference()
        losses = trainer.get_current_errors()

        loss_total.append(losses['loss_total'].cpu().item())
        loss_idt.append(losses['idt_S'].cpu().item() + losses['idt_R'].cpu().item())
        loss_ssim.append(losses['ssim_S'].cpu().item() + losses['ssim_R'].cpu().item())
        loss_grad.append(losses['grad_S'].cpu().item() + losses['grad_R'].cpu().item())
        loss_fea_diver.append(losses['fea_divergence'].cpu().item())
        loss_perspective.append(losses['perspective'].cpu().item())
        loss_fea_extract.append(losses['fea_extract'].cpu().item())
        loss_preserve_info.append(losses['preserve_info'].cpu().item())

        if opt.train.trainer_mode == 'GAN':
            loss_GAN_G.append(losses['gan_G'].cpu().item())
            loss_GAN_D_s.append(losses['gan_D_s'].cpu().item())
            loss_GAN_D_r.append(losses['gan_D_r'].cpu().item())

        # print eval losses
        log_str = 'Eval Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx, batch_num,
            100.0 * batch_idx / batch_num, losses['loss_total'].cpu().item())
        if batch_idx % 100 == 0:
            print(log_str)
        log_detail = trainer.loss_log(losses)
        # print(log_detail)
        log_list.append(log_str)
        log_list.append(log_detail)

        if save_pred_results:
            # save eval imgs into epoch dir
            # if batch_idx % 10 == 0:
            #     print(log_str)
            visuals = trainer.get_current_visuals()
            img_dir = opt.logger.log_dir + 'test-imgs_ep' + str(epoch)  # '../checkpoints/log/'
            check_dir(img_dir)
            save_eval_images(visuals, img_dir, batch_idx, opt)

    loss_inspector['loss_total'].append(np.mean(loss_total))
    loss_inspector['loss_idt'].append(np.mean(loss_idt))
    loss_inspector['loss_ssim'].append(np.mean(loss_ssim))
    loss_inspector['loss_grad'].append(np.mean(loss_grad))
    loss_inspector['loss_fea_diver'].append(np.mean(loss_fea_diver))
    loss_inspector['loss_perspective'].append(np.mean(loss_perspective))
    loss_inspector['loss_fea_extract'].append(np.mean(loss_fea_extract))
    loss_inspector['loss_preserve_info'].append(np.mean(loss_preserve_info))
    cur_lr = opt.optim.lr_g if trainer.get_lr() is None else trainer.get_lr()
    loss_inspector['lr'].append(cur_lr)

    if opt.train.trainer_mode == 'GAN':
        loss_inspector['loss_GAN_G'].append(np.mean(loss_GAN_G))
        loss_inspector['loss_GAN_D_s'].append(np.mean(loss_GAN_D_s))
        loss_inspector['loss_GAN_D_r'].append(np.mean(loss_GAN_D_r))

    loss_inspector['step'].append(epoch)
    # save log info into file
    #pdb.set_trace()
    eval_log = 'Evaluation_loss_total-Ep{}: {:.4f}, learning rate: {}\nloss_idt: {:.4f},\tloss_ssim: \
    {:.4f}\tloss_grad: {:.4f} \tloss_fea_diver: {:.4f},\tloss_perspective:\
    {:.4f},\tloss_fea_extract: {:.4f}, \tloss_preserve_info:{:.4f}\n'.format(
               epoch, np.mean(loss_total), trainer.get_lr(), np.mean(loss_idt), np.mean(loss_ssim), np.mean(loss_grad),
               np.mean(loss_fea_diver), np.mean(loss_perspective),
               np.mean(loss_fea_extract), np.mean(loss_preserve_info))
    if opt.train.trainer_mode == 'GAN':
        eval_log = 'Evaluation_loss_total-Ep{}: {:.4f},\nloss_idt: {:.4f},\tloss_ssim: \
            {:.4f}\tloss_grad: {:.4f} \tloss_fea_diver: {:.4f},\tloss_perspective: {:.4f},\tloss_fea_extract: {:.4f}\n\
            loss_GAN_G: {:.4f} \tloss_GAN_D_s: {:.4f} \tloss_GAN_D_r: {:.4f}'.format(
            epoch, np.mean(loss_total), np.mean(loss_idt), np.mean(loss_ssim), np.mean(loss_grad),
            np.mean(loss_fea_diver), np.mean(loss_perspective), np.mean(loss_fea_extract),
            np.mean(loss_GAN_G), np.mean(loss_GAN_D_s), np.mean(loss_GAN_D_r))
    print(eval_log)
    if save_pred_results:
        with open(opt.logger.log_dir + '/eval_loss_log-ep' + str(epoch) + '.txt', 'w') as f:
            f.write(eval_log)
            f.writelines(["%s\n" % item for item in log_list])


def basic_settings():
    run_settings = {}
    """setup datasets"""
    if 'MPI' in opt.data.name:
        train_dataset = my_data.DatasetIdMPI_mask(opt.data, is_train=True, cropped=False)
        test_dataset = my_data.DatasetIdMPI_mask(opt.data, is_train=False, cropped=False)
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


def train():
    s = basic_settings()

    start_epoch = opt.train.epoch_count  # 1

    if opt.continue_train:
        evaluate_one_epoch(s['model_trainer'], s['test_dataset'], start_epoch-1, s['test_loss_inspector'])
    for ep in range(opt.train.total_ep):  # total_ep=300
        epoch = ep + start_epoch
        s['model_trainer'].update_learning_rate()
        s['model_trainer'].update_optim_weights(ep)
        train_one_epoch(s['model_trainer'], s['train_dataset'], epoch, s['train_loss_inspector'])
        if epoch % opt.train.save_per_n_ep == 0:
            evaluate_one_epoch(s['model_trainer'], s['test_dataset'], epoch, s['test_loss_inspector'])
        if epoch % opt.train.save_per_n_ep == 0:
            s['model_trainer'].save(epoch)

            """save to log"""
            if opt.train.trainer_mode == 'Basic':
                with open(opt.logger.log_dir + '/all_eval_log-' + str(start_epoch - 1 + opt.train.total_ep) + '.txt', 'w') as f:
                    f.write(
                        'epoch: total_loss: idt_loss: ssim_loss: grad_loss:\
                        fea_divergence: perspective: fea_extract:\
                        learning_rate:\n')
                    for i in range(len(s['test_loss_inspector']['step'])):
                        log_string = '{:04d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.7f}\n'.format(
                            s['test_loss_inspector']['step'][i],
                            s['test_loss_inspector']['loss_total'][i],
                            s['test_loss_inspector']['loss_idt'][i],
                            s['test_loss_inspector']['loss_ssim'][i],
                            s['test_loss_inspector']['loss_grad'][i],
                            s['test_loss_inspector']['loss_fea_diver'][i],
                            s['test_loss_inspector']['loss_perspective'][i],
                            s['test_loss_inspector']['loss_fea_extract'][i],
                            s['test_loss_inspector']['lr'][i]
                        )
                        f.write(log_string)
            elif opt.train.trainer_mode == 'GAN':
                with open(opt.logger.log_dir + '/all_eval_log-' + str(start_epoch-1+opt.train.total_ep) + '.txt', 'w') as f:
                    f.write('epoch: total_loss: idt_loss: ssim_loss: grad_loss: fea_divergence: perspective: fea_extract\
                             GAN_G: GAN_D_s: GAN_D_r:\n')
                    for i in range(len(s['test_loss_inspector']['step'])):
                        log_string = '{:04d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(
                            s['test_loss_inspector']['step'][i],
                            s['test_loss_inspector']['loss_total'][i],
                            s['test_loss_inspector']['loss_idt'][i],
                            s['test_loss_inspector']['loss_ssim'][i],
                            s['test_loss_inspector']['loss_grad'][i],
                            s['test_loss_inspector']['loss_fea_diver'][i],
                            s['test_loss_inspector']['loss_perspective'][i],
                            s['test_loss_inspector']['loss_fea_extract'][i],
                            s['test_loss_inspector']['loss_GAN_G'][i],
                            s['test_loss_inspector']['loss_GAN_D_s'][i],
                            s['test_loss_inspector']['loss_GAN_D_r'][i]
                        )
                        f.write(log_string)
            else:
                raise NotImplementedError
            """visualize inspectors"""
            train_loss_dir = opt.logger.log_dir + 'train_losses-' + \
                             str(start_epoch) + '-' + str(epoch) + '_' + str(start_epoch - 1 + opt.train.total_ep) + '.png'
            test_loss_dir = opt.logger.log_dir + 'test_losses-' + \
                            str(start_epoch) + '-' + str(epoch) + '_' + str(start_epoch - 1 + opt.train.total_ep) + '.png'
            visualize_inspector(s['train_loss_inspector'], train_loss_dir, step_num=None, mode=opt.train.trainer_mode)
            visualize_inspector(s['test_loss_inspector'], test_loss_dir, step_num=None, mode=opt.train.trainer_mode)


def test(epoch=None):
    settings = basic_settings()
    test_dataset = settings['test_dataset']
    model_trainer = settings['model_trainer']
    test_loss_inspector = settings['test_loss_inspector']

    model_trainer.resume(model_trainer.gen_split, 'G_decompose', epoch_name=epoch)
    if opt.train.trainer_mode == 'GAN':
        model_trainer.resume(model_trainer.dis_S, 'D_S', epoch_name=epoch)
        model_trainer.resume(model_trainer.dis_R, 'D_R', epoch_name=epoch)

    evaluate_one_epoch(model_trainer, test_dataset, epoch=epoch,
                       loss_inspector=test_loss_inspector, save_pred_results=True)


if __name__ == '__main__':
    #train()
    test(epoch=185)
    #test(epoch=85)
    #test(epoch=150)

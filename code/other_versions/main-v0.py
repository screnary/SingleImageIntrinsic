import torch
import torchvision
from torch.utils.data import DataLoader
import data
from configs.intrinsic import opt
import trainer as Trainer
from utils import save_eval_images, check_dir, visualize_inspector
import numpy as np
import pdb

check_dir(opt.output_root)
check_dir(opt.logger.log_dir)


# loss_inspector = {'loss_total': [], 'loss_idt': [], 'loss_ssim': [], 'loss_grad': [], 'iter_step': []}
def train_one_epoch(trainer, dataset, epoch, loss_inspector):
    opt.is_train = True
    dataloader = DataLoader(dataset, batch_size=opt.data.batch_size, shuffle=True)
    batch_num = len(dataset)//opt.data.batch_size

    log_list = []

    for batch_idx, samples in enumerate(dataloader):
        trainer.set_input(samples)
        trainer.optimize_parameters()

        losses = trainer.get_current_errors()

        loss_inspector['loss_total'].append(losses['loss_total'].cpu().item())
        loss_inspector['loss_idt'].append(losses['idt_S'].cpu().item() + losses['idt_R'].cpu().item())
        loss_inspector['loss_ssim'].append(losses['ssim_S'].cpu().item() + losses['ssim_R'].cpu().item())
        loss_inspector['loss_grad'].append(losses['grad_S'].cpu().item() + losses['grad_R'].cpu().item())
        loss_inspector['step'].append((epoch-1) * batch_num + batch_idx + 1)

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, batch_num,
                100.0*batch_idx*batch_num/len(dataset), losses['loss_total']))
            log_detail = trainer.loss_log(losses)
            print(log_detail)
            log_list.append(log_detail)
        if opt.train.save_train_img and (epoch % opt.train.save_per_n_ep) == 0:
            visuals = trainer.get_current_visuals()
            img_dir = opt.logger.log_dir + 'train-imgs_ep' + str(epoch)  # '../checkpoints/log/'
            check_dir(img_dir)
            save_eval_images(visuals, img_dir, batch_idx, opt)
    if epoch % opt.train.save_per_n_ep == 0:
        with open(opt.logger.log_dir + '/train_loss_log-ep' + str(epoch) + '.txt', 'w') as f:
            f.writelines(["%s\n" % item for item in log_list])


def evaluate(trainer, dataset, epoch, loss_inspector):
    opt.is_train = False
    # Todo:batch_size should be 1, or change save_images function
    batch_size = opt.data.batch_size_test
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch_num = len(dataset) // batch_size

    log_list = []
    loss_total = []
    loss_idt = []
    loss_ssim = []
    loss_grad = []
    for batch_idx, samples in enumerate(dataloader):
        trainer.set_input(samples)
        trainer.inference()
        losses = trainer.get_current_errors()

        loss_total.append(losses['loss_total'].cpu().item())
        loss_idt.append(losses['idt_S'].cpu().item() + losses['idt_R'].cpu().item())
        loss_ssim.append(losses['ssim_S'].cpu().item() + losses['ssim_R'].cpu().item())
        loss_grad.append(losses['grad_S'].cpu().item() + losses['grad_R'].cpu().item())

        # print eval losses
        log_str = 'Eval Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx, batch_num,
            100.0 * batch_idx * batch_num / len(dataset), losses['loss_total'])
        print(log_str)
        log_detail = trainer.loss_log(losses)
        print(log_detail)
        log_list.append(log_str)
        log_list.append(log_detail)

        if epoch % opt.train.save_per_n_ep == 0:
            # save eval imgs into epoch dir
            visuals = trainer.get_current_visuals()
            img_dir = opt.logger.log_dir + 'test-imgs_ep' + str(epoch)  # '../checkpoints/log/'
            check_dir(img_dir)
            save_eval_images(visuals, img_dir, batch_idx, opt)

    loss_inspector['loss_total'].append(np.mean(loss_total))
    loss_inspector['loss_idt'].append(np.mean(loss_idt))
    loss_inspector['loss_ssim'].append(np.mean(loss_ssim))
    loss_inspector['loss_grad'].append(np.mean(loss_grad))
    loss_inspector['step'].append(epoch)
    # save log info into file
    with open(opt.logger.log_dir + '/eval_loss_log-ep' + str(epoch) + '.txt', 'w') as f:
        f.write('loss_total: {:.6f},\nloss_idt: {:.6f},\tloss_ssim: {:.6f}\tloss_grad: {:.6f}\n'.format(
            np.mean(loss_total), np.mean(loss_idt), np.mean(loss_ssim), np.mean(loss_grad)
        ))
        f.writelines(["%s\n" % item for item in log_list])


if __name__ == '__main__':
    """setup datasets"""
    if 'MPI' in opt.data.name:
        train_dataset = data.DatasetIdMPI(opt.data, is_train=True)
        test_dataset = data.DatasetIdMPI(opt.data, is_train=False)
    elif 'MIT' in opt.data.name:
        train_dataset = data.DatasetIdMIT(opt.data, is_train=True)
        test_dataset = data.DatasetIdMIT(opt.data, is_train=False)
    else:
        raise NotImplementedError

    """setup model trainer"""
    if opt.train.trainer_mode == 'GAN':
        model_trainer = Trainer.Trainer_GAN(opt)
    else:
        model_trainer = Trainer.Trainer_Basic(opt)

    # evaluate(trainer, test_dataset, 0)
    # pdb.set_trace()
    """training process"""
    train_loss_inspector = {'loss_total': [], 'loss_idt': [], 'loss_ssim': [], 'loss_grad': [], 'step': []}
    test_loss_inspector = {'loss_total': [], 'loss_idt': [], 'loss_ssim': [], 'loss_grad': [], 'step': []}

    start_epoch = opt.train.epoch_count  # 1
    for ep in range(opt.train.total_ep):  # total_ep=300
        epoch = ep + start_epoch
        model_trainer.update_learning_rate()
        train_one_epoch(model_trainer, train_dataset, epoch, train_loss_inspector)
        evaluate(model_trainer, test_dataset, epoch, test_loss_inspector)
        if epoch % opt.train.save_per_n_ep == 0:
            model_trainer.save(epoch)

    """visualize inspectors"""
    train_loss_dir = opt.logger.log_dir + 'train_losses.png'
    test_loss_dir = opt.logger.log_dir + 'test_losses.png'
    visualize_inspector(train_loss_inspector, train_loss_dir, step_num=None)
    visualize_inspector(test_loss_inspector, test_loss_dir, step_num=None)

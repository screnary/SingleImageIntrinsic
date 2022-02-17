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
        loss_inspector['loss_fea_diver'].append(losses['fea_divergence'].cpu().item())
        loss_inspector['loss_perspective'].append(losses['perspective'].cpu().item())
        loss_inspector['loss_fea_extract'].append(losses['fea_extract'].cpu().item())

        loss_inspector['step'].append((epoch-1) * batch_num + batch_idx + 1)

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, batch_num,
                100.0*batch_idx/batch_num, losses['loss_total'].cpu().item()))
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

        # print eval losses
        log_str = 'Eval Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx, batch_num,
            100.0 * batch_idx / batch_num, losses['loss_total'].cpu().item())
        # print(log_str)
        log_detail = trainer.loss_log(losses)
        # print(log_detail)
        log_list.append(log_str)
        log_list.append(log_detail)

        if save_pred_results:
            # save eval imgs into epoch dir
            if batch_idx % 10 == 0:
                print(log_str)
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

    loss_inspector['step'].append(epoch)
    # save log info into file
    eval_log = 'Evaluation_loss_total-Ep{}: {:.4f},\nloss_idt: {:.4f},\tloss_ssim: \
    {:.4f}\tloss_grad: {:.4f} \tloss_fea_diver: {:.4f},\tloss_perspective: {:.4f},\tloss_fea_extract: {:.4f}\n'.format(
               epoch, np.mean(loss_total), np.mean(loss_idt), np.mean(loss_ssim), np.mean(loss_grad),
               np.mean(loss_fea_diver), np.mean(loss_perspective), np.mean(loss_fea_extract))
    print(eval_log)
    if save_pred_results:
        with open(opt.logger.log_dir + '/eval_loss_log-ep' + str(epoch) + '.txt', 'w') as f:
            f.write(eval_log)
            f.writelines(["%s\n" % item for item in log_list])


def basic_settings():
    run_settings = {}
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
    train_loss_inspector = {'loss_total': [], 'loss_idt': [], 'loss_ssim': [], 'loss_grad': [], 'loss_fea_diver': [],
                            'loss_perspective': [], 'loss_fea_extract': [], 'step': []}
    test_loss_inspector = {'loss_total': [], 'loss_idt': [], 'loss_ssim': [], 'loss_grad': [], 'loss_fea_diver': [],
                           'loss_perspective': [], 'loss_fea_extract': [], 'step': []}

    run_settings['train_dataset'] = train_dataset
    run_settings['test_dataset'] = test_dataset
    run_settings['model_trainer'] = model_trainer
    run_settings['train_loss_inspector'] = train_loss_inspector
    run_settings['test_loss_inspector'] = test_loss_inspector

    return run_settings


def train():
    s = basic_settings()

    start_epoch = opt.train.epoch_count  # 1
    for ep in range(opt.train.total_ep):  # total_ep=300
        epoch = ep + start_epoch
        s['model_trainer'].update_learning_rate()
        train_one_epoch(s['model_trainer'], s['train_dataset'], epoch, s['train_loss_inspector'])
        evaluate_one_epoch(s['model_trainer'], s['test_dataset'], epoch, s['test_loss_inspector'])
        if epoch % opt.train.save_per_n_ep == 0:
            s['model_trainer'].save(epoch)

    """save to log"""
    with open(opt.logger.log_dir + '/all_eval_log-' + str(start_epoch-1+opt.train.total_ep) + '.txt', 'w') as f:
        f.write('epoch:\t total_loss:\t idt_loss:\t ssim_loss:\t grad_loss:\t fea_divergence:\t perspective:\t fea_extract\n')
        for i in range(len(s['test_loss_inspector']['step'])):
            log_string = '{:04d}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n'.format(
                s['test_loss_inspector']['step'][i],
                s['test_loss_inspector']['loss_total'][i],
                s['test_loss_inspector']['loss_idt'][i],
                s['test_loss_inspector']['loss_ssim'][i],
                s['test_loss_inspector']['loss_grad'][i],
                s['test_loss_inspector']['loss_fea_diver'][i],
                s['test_loss_inspector']['loss_perspective'][i],
                s['test_loss_inspector']['loss_fea_extract'][i],
            )
            f.write(log_string)
    """visualize inspectors"""
    train_loss_dir = opt.logger.log_dir + 'train_losses-' + \
                     str(start_epoch) + '_' + str(start_epoch - 1 + opt.train.total_ep) + '.png'
    test_loss_dir = opt.logger.log_dir + 'test_losses-' + \
                    str(start_epoch) + '_' + str(start_epoch - 1 + opt.train.total_ep) + '.png'
    visualize_inspector(s['train_loss_inspector'], train_loss_dir, step_num=None)
    visualize_inspector(s['test_loss_inspector'], test_loss_dir, step_num=None)


def test(epoch=None):
    settings = basic_settings()
    test_dataset = settings['test_dataset']
    model_trainer = settings['model_trainer']
    test_loss_inspector = settings['test_loss_inspector']

    model_trainer.resume(model_trainer.gen_split, 'G_decompose', epoch_name=epoch)
    evaluate_one_epoch(model_trainer, test_dataset, epoch=epoch,
                       loss_inspector=test_loss_inspector, save_pred_results=True)


if __name__ == '__main__':
    # train()
    test(epoch=100)

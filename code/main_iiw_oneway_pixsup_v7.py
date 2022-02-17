import torch
import torchvision
import my_data
#from configs.intrinsic_iiw import opt
import argparse
from configs.intrinsic_iiw_oneway_pixsup_v7 import opt
from utils import save_eval_images_iiw, check_dir, visualize_inspector_iiw

import os
import numpy as np

import pdb
import warnings
warnings.filterwarnings('ignore')

# >>>>>> use this for iiw dataset results <<<<<<
# >>>>>>> [2021.03.08] >>>>>>>
# use this as base, change the weights of smoothness priors


######## argparse for flexible change variables #####
#### need to define:
# opt.optim.w_rs_local = 1.0                  # multiscale smooth for reflectance [1.0]
# opt.optim.w_ss_dense = 1.0                  # dense shading smooth loss         [4.0]

# opt.output_root = '../ckpoints-' + opt.train.trainer_mode + '-' + opt.train.mode + '-' + opt.data.name \
                    #   + '-decoder_' + opt.model.gen.decoder_mode
# opt.logger.log_dir = opt.output_root+'/log/'
########

parser = argparse.ArgumentParser(description="ablation study of loss terms for IIW dataset")
parser.add_argument('--w_rs', type=float, default=0.0)
parser.add_argument('--w_ss', type=float, default=0.0)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--continue_train', action='store_true')
parser.add_argument('--which_epoch', type=int, default=None)
parser.add_argument('--phase', type=str, default='train', help="train or test")
parser.add_argument('--best_ep', type=int, default=None)
FLAGS = parser.parse_args()
# pdb.set_trace()
opt.continue_train = FLAGS.continue_train
opt.which_epoch= FLAGS.which_epoch
opt.train.epoch_count = 1 if not opt.continue_train else opt.which_epoch+1    # the starting epoch count: epoch_count += save_freq

opt.optim.w_rs_local = FLAGS.w_rs
opt.optim.w_ss_dense = FLAGS.w_ss
opt.output_root = '../ckpoints-'+opt.train.mode+'-'+opt.data.name+'-'\
                  + '-w_rs-' + str(opt.optim.w_rs_local) + '-w_ss-' + str(opt.optim.w_ss_dense)
opt.logger.log_dir = opt.output_root + '/log/'

check_dir(opt.output_root)
check_dir(opt.logger.log_dir)

opt.gpu_ids = [FLAGS.gpu_id]
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu_id)
print('cuda visible: ', os.environ["CUDA_VISIBLE_DEVICES"])
#<<<<<<< 2021.03.08 <<<<<<<

import trainer_iiw_new as Trainer
from utils.image_pool import ImagePool

# loss_inspector = {'loss_total': [], 'loss_idt': [], 'loss_ssim': [], 'loss_grad': [], 'iter_step': []}
def train_one_epoch(trainer, epoch, loss_inspector):
    opt.is_train = True
    img_num = 0

    for j in range(0, 5):
        batch_size = opt.data.batch_size
        if j > 1 and opt.data.batch_size > 1:
            batch_size = opt.data.batch_size // 2
        dataset = my_data.DatasetIdIIW(opt.data, is_train=True, mode=j, batch_size=batch_size)
        dataloader = dataset.load()
        batch_num = len(dataset)//batch_size

        log_list = []
        trainer.fake_R_pool = ImagePool(opt.train.pool_size)  # reset image pool for new image height and width
        trainer.fake_S_pool = ImagePool(opt.train.pool_size)

        for batch_idx, samples in enumerate(dataloader):
            stacked_img = samples['img_1']
            targets = samples['target_1']
            trainer.set_input(stacked_img, targets)
            trainer.optimize_parameters()

            losses = trainer.get_current_errors()
            if np.isnan(losses['loss_total']):
                print('Warning: nan loss!')
                pdb.set_trace()

            total_whdr, total_whdr_eq, total_whdr_ineq, count = trainer.evlaute_iiw()

            loss_inspector['loss_total'].append(losses['loss_total'])  # .cpu().item()
            loss_inspector['loss_whdr'].append(total_whdr / count)
            loss_inspector['loss_rs'].append(losses['loss_rs'])
            loss_inspector['loss_ss'].append(losses['loss_ss'])
            loss_inspector['loss_iiw'].append(losses['loss_iiw'])
            loss_inspector['fea_divergence'].append(losses['fea_divergence'])
            loss_inspector['perceptive'].append(losses['perceptive'])
            loss_inspector['preserve_info'].append(losses['preserve_info'])
            loss_inspector['pixel_penalty'].append(losses['pixel_penalty'])
            loss_inspector['lr'].append(trainer.get_lr())

            cur_step = loss_inspector['step'][-1] if len(loss_inspector['step']) > 0 else 0
            loss_inspector['step'].append(cur_step+1)

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]-orient:{} \tLoss: {:.6f}  loss_whdr: {:.5f}'.format(
                    epoch, batch_idx, batch_num,
                    100.0*batch_idx/batch_num, j, losses['loss_total'], total_whdr/count))  # .cpu().item()
                log_detail = trainer.loss_log(losses)
                print(log_detail)
                log_list.append(log_detail)
            if opt.train.save_train_img and (epoch % opt.train.save_per_n_ep_train) == 0 and img_num % 100 == 0:
                visuals = trainer.get_current_visuals()
                img_dir = opt.logger.log_dir + 'train-imgs_view'  # '../checkpoints/log/'
                check_dir(img_dir)
                save_eval_images_iiw(visuals, img_dir, 'ep-'+str(epoch)+'_'+str(loss_inspector['step'][-1]), opt)

            img_num += 1

        if epoch % opt.train.save_per_n_ep == 0:
            with open(opt.logger.log_dir + '/train_loss_log-ep' + str(epoch) + '.txt', 'w') as f:
                f.writelines(["%s\n" % item for item in log_list])


def evaluate_one_epoch(trainer, epoch, loss_inspector, save_pred_results=False):
    opt.is_train = False
    # Todo:batch_size should be 1, or change save_images function
    batch_size = opt.data.batch_size_test if not save_pred_results else 1

    log_list = []
    loss_total = []

    total_loss_whdr = 0.0
    total_loss_whdr_eq = 0.0
    total_loss_whdr_ineq = 0.0
    total_count = 0.0
    img_num = 0

    for j in range(0, 3):
        if not save_pred_results and j > 1 and batch_size > 1:
            batch_size = opt.data.batch_size_test // 2
        dataset = my_data.DatasetIdIIW(opt.data, is_train=False, mode=j, batch_size=batch_size)
        dataloader = dataset.load()
        batch_num = max(int(1), len(dataset) // batch_size)

        for batch_idx, samples in enumerate(dataloader):
            stacked_img = samples['img_1']
            targets = samples['target_1']
            trainer.set_input(stacked_img, targets)
            trainer.inference()
            total_whdr, total_whdr_eq, total_whdr_ineq, count = trainer.evlaute_iiw()

            total_loss_whdr += total_whdr
            total_loss_whdr_eq += total_whdr_eq
            total_loss_whdr_ineq += total_whdr_ineq
            total_count += count

            # print eval losses
            log_str = 'Eval Epoch: {} [{}/{} ({:.0f}%)]-orient:{} \t\
                       img_num:{}  loss_whdr:{:.5f}\n'.format(
                epoch, batch_idx, batch_num,
                100.0 * batch_idx / batch_num, j, img_num, total_whdr / count)
            if batch_idx % 100 == 0:
                print(log_str)
            # print(log_detail)
            log_list.append(log_str)

            if save_pred_results:
                # save eval imgs into epoch dir
                # if batch_idx % 10 == 0:
                #     print(log_str)
                visuals = trainer.get_current_visuals()
                img_dir = opt.logger.log_dir + 'test-imgs_ep' + str(epoch)  # '../checkpoints/log/'
                check_dir(img_dir)
                save_eval_images_iiw(visuals, img_dir, str(img_num), opt)
            img_num += 1

    loss_inspector['loss_whdr'].append(total_loss_whdr / total_count)
    loss_inspector['step'].append(epoch)

    cur_lr = opt.optim.lr_g if trainer.get_lr() is None else trainer.get_lr()
    loss_inspector['lr'].append(cur_lr)

    # save log info into file
    eval_log = 'Evaluation_loss_total-Ep{}: loss_whdr: {:.6f}\n'.format(
               epoch, total_loss_whdr / total_count)
    print(eval_log)
    if save_pred_results:
        with open(opt.logger.log_dir + '/eval_loss_log-ep' + str(epoch) + '.txt', 'w') as f:
            f.write(eval_log)
            f.writelines(["%s\n" % item for item in log_list])


def check_one_epoch(trainer, epoch, loss_inspector, save_pred_results=False):
    opt.is_train = False
    # Todo:batch_size should be 1, or change save_images function
    # batch_size = opt.data.batch_size_test if not save_pred_results else 1
    batch_size = 4

    log_list = []
    loss_total = []

    total_loss_whdr = 0.0
    total_loss_whdr_eq = 0.0
    total_loss_whdr_ineq = 0.0
    total_count = 0.0
    img_num = 0

    for j in range(0, 1):
        if not save_pred_results and j > 1 and batch_size > 1:
            batch_size = opt.data.batch_size_test // 2
        dataset = my_data.DatasetIdIIW(opt.data, is_train=True, mode=j, batch_size=batch_size, is_check=True)
        dataloader = dataset.load()
        batch_num = max(int(1), len(dataset) // batch_size)

        for batch_idx, samples in enumerate(dataloader):
            stacked_img = samples['img_1']
            targets = samples['target_1']
            trainer.set_input(stacked_img, targets)
            trainer.inference()
            total_whdr, total_whdr_eq, total_whdr_ineq, count = trainer.evlaute_iiw()

            total_loss_whdr += total_whdr
            total_loss_whdr_eq += total_whdr_eq
            total_loss_whdr_ineq += total_whdr_ineq
            total_count += count

            # print eval losses
            log_str = 'Check Epoch: {} [{}/{} ({:.0f}%)]-orient:{} \t\
                       img_num:{}  loss_whdr:{:.5f}\n'.format(
                epoch, batch_idx, batch_num,
                100.0 * batch_idx / batch_num, j, img_num, total_whdr / count)
            if batch_idx % 100 == 0:
                print(log_str)
            # print(log_detail)
            log_list.append(log_str)

            if save_pred_results and img_num % 100 == 0:
                # save eval imgs into epoch dir
                # if batch_idx % 10 == 0:
                #     print(log_str)
                visuals = trainer.get_current_visuals()
                img_dir = opt.logger.log_dir + 'check-imgs_ep' + str(epoch)  # '../checkpoints/log/'
                check_dir(img_dir)
                save_eval_images_iiw(visuals, img_dir, str(img_num), opt)
            img_num += batch_size

    loss_inspector['loss_whdr'].append(total_loss_whdr / total_count)
    loss_inspector['step'].append(epoch)
    # save log info into file
    eval_log = 'Evaluation_loss_total-Ep{}: loss_whdr: {:.6f}\n'.format(
               epoch, total_loss_whdr / total_count)
    print(eval_log)
    if save_pred_results:
        with open(opt.logger.log_dir + '/check_loss_log-ep' + str(epoch) + '.txt', 'w') as f:
            f.write(eval_log)
            f.writelines(["%s\n" % item for item in log_list])


def basic_settings():
    run_settings = {}

    """setup model trainer"""
    model_trainer = Trainer.Trainer_Basic(opt)

    # evaluate(trainer, test_dataset, 0)
    # pdb.set_trace()
    """training process"""
    train_loss_inspector = {'loss_total': [], 'loss_whdr':[], 'loss_idt': [], 'loss_ssim': [], 'loss_grad': [], 
                            'loss_rs': [], 'loss_ss': [], 'loss_iiw': [],
                            'fea_divergence': [], 'perceptive': [], 'lr': [],
                            'preserve_info': [], 'pixel_penalty': [],
                            'loss_fea_diver': [], 'loss_perspective': [], 'loss_fea_extract': [], 'step': []}
    test_loss_inspector = {'loss_total': [], 'loss_whdr': [], 'loss_idt': [], 'loss_ssim': [], 'loss_grad': [], 
                           'loss_rs': [], 'loss_ss': [], 'loss_iiw': [],
                           'fea_divergence': [], 'perceptive': [], 'lr': [],
                           'preserve_info': [], 'pixel_penalty': [],
                           'loss_fea_diver': [], 'loss_perspective': [], 'loss_fea_extract': [], 'step': []}

    run_settings['model_trainer'] = model_trainer
    run_settings['train_loss_inspector'] = train_loss_inspector
    run_settings['test_loss_inspector'] = test_loss_inspector

    return run_settings


def train():
    s = basic_settings()

    start_epoch = opt.train.epoch_count  # 1

    if opt.continue_train:
        print("continue_train! start epoch id is: ", start_epoch)
        evaluate_one_epoch(s['model_trainer'], start_epoch-1, s['test_loss_inspector'])

    for ep in range(opt.train.total_ep):  # total_ep=300
        epoch = ep + start_epoch
        s['model_trainer'].update_learning_rate()
        if ep > 0:
            s['model_trainer'].update_threshold()
        train_one_epoch(s['model_trainer'], epoch, s['train_loss_inspector'])
        if epoch % opt.train.save_per_n_ep == 0:
            evaluate_one_epoch(s['model_trainer'], epoch, s['test_loss_inspector'])
        if epoch % opt.train.save_per_n_ep == 0:
            s['model_trainer'].save(epoch)

            """save to log"""
            with open(opt.logger.log_dir + '/all_eval_log-' + str(start_epoch - 1 + opt.train.total_ep) + '.txt', 'w') as f:
                f.write('epoch: whdr_loss\n')
                for i in range(len(s['test_loss_inspector']['step'])):
                    log_string = '{:04d}\t{:.4f}\n'.format(
                        s['test_loss_inspector']['step'][i],
                        s['test_loss_inspector']['loss_whdr'][i],
                    )
                    f.write(log_string)

            """visualize inspectors"""
            train_loss_dir = opt.logger.log_dir + 'train_losses-' + \
                             str(start_epoch) + '_' + str(epoch) + '-' + str(start_epoch - 1 + opt.train.total_ep) + '.png'
            test_loss_dir = opt.logger.log_dir + 'test_losses-' + \
                            str(start_epoch) + '_' + str(epoch) + '-' + str(start_epoch - 1 + opt.train.total_ep) + '.png'
            visualize_inspector_iiw(s['train_loss_inspector'], train_loss_dir, step_num=None, mode=opt.train.trainer_mode)
            visualize_inspector_iiw(s['test_loss_inspector'], test_loss_dir, step_num=None, mode=opt.train.trainer_mode)

    eval_whdr = np.asarray(s['test_loss_inspector']['loss_whdr'])
    ind = np.argmin(eval_whdr)
    best_whdr = eval_whdr[ind]
    best_epoch = s['test_loss_inspector']['step'][ind]
    return best_whdr, best_epoch


def test(epoch=None):
    settings = basic_settings()
    model_trainer = settings['model_trainer']
    test_loss_inspector = settings['test_loss_inspector']

    model_trainer.resume(model_trainer.gen_decompose, 'G_decompose', epoch_name=epoch)

    evaluate_one_epoch(model_trainer, epoch=epoch,
                       loss_inspector=test_loss_inspector, save_pred_results=True)


def check(epoch=None):
    """check performance on training set"""
    settings = basic_settings()
    model_trainer = settings['model_trainer']
    test_loss_inspector = settings['test_loss_inspector']

    model_trainer.resume(model_trainer.gen_decompose, 'G_decompose', epoch_name=epoch)

    check_one_epoch(model_trainer, epoch=epoch,
                    loss_inspector=test_loss_inspector, save_pred_results=True)


if __name__ == '__main__':
    assert torch.cuda.is_available()
    if FLAGS.phase=="train":
        best_whdr, best_epoch = train()
        print('whdr=', best_whdr)
        print('best ep:', best_epoch)
    if FLAGS.phase=="test":
        if FLAGS.best_ep is not None:
            test(epoch=FLAGS.best_ep)
        else:
            test(epoch=10)  # use this finally

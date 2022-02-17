import torch
import torchvision
import my_data
#from configs.intrinsic_iiw import opt
from configs.intrinsic_iiw_v3 import opt
import trainer_iiw as Trainer
from utils import save_eval_images_iiw, check_dir, visualize_inspector_iiw
from utils.image_pool import ImagePool
import numpy as np
import pdb
import warnings
warnings.filterwarnings('ignore')

check_dir(opt.output_root)
check_dir(opt.logger.log_dir)


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

            total_whdr, total_whdr_eq, total_whdr_ineq, count = trainer.evlaute_iiw()

            loss_inspector['loss_total'].append(losses['loss_total'])  # .cpu().item()
            loss_inspector['loss_whdr'].append(total_whdr / count)
            cur_step = loss_inspector['step'][-1] if len(loss_inspector['step']) > 0 else 0
            loss_inspector['step'].append(cur_step+1)

            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]-orient:{} \tLoss: {:.6f}  loss_whdr: {:.5f}'.format(
                    epoch, batch_idx, batch_num,
                    100.0*batch_idx/batch_num, j, losses['loss_total'], total_whdr/count))  # .cpu().item()
                log_detail = trainer.loss_log(losses)
                print(log_detail)
                log_list.append(log_detail)
            if opt.train.save_train_img and (epoch % opt.train.save_per_n_ep) == 0 and img_num % 100 == 0:
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
    train_loss_inspector = {'loss_total': [], 'loss_whdr':[], 'loss_idt': [], 'loss_ssim': [], 'loss_grad': [], 'loss_fea_diver': [],
                            'loss_perspective': [], 'loss_fea_extract': [], 'step': []}
    test_loss_inspector = {'loss_total': [], 'loss_whdr': [], 'loss_idt': [], 'loss_ssim': [], 'loss_grad': [], 'loss_fea_diver': [],
                           'loss_perspective': [], 'loss_fea_extract': [], 'step': []}

    run_settings['model_trainer'] = model_trainer
    run_settings['train_loss_inspector'] = train_loss_inspector
    run_settings['test_loss_inspector'] = test_loss_inspector

    return run_settings


def train():
    s = basic_settings()

    start_epoch = opt.train.epoch_count  # 1

    if opt.continue_train:
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
    train()
    # check(epoch=10)
    #test(epoch=14)
    # test(epoch=15)
    # test(epoch=20)

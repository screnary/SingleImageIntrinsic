import os
import shutil
import time

import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib as mpl
mpl.use('Agg')  # force matplotlib to not use any Xwindows backend
import matplotlib.pyplot as plt
import pdb

"""
@tensor2img                 :   convert Variable or tensor into image with un-normalized operation
@show_image                 :   show an image with opencv
@check_dir                  :   if a dir is not exist, create it
@remove_dir                 :   if a dir is exist, remove it
@get_dataloader
@get_scheduler
@print_networ
@weights_init
"""


def visualize_inspector(inspector, save_dir, step_num=None, mode='Basic'):
    fig_n = 3  # subplot blocks number
    # loss_inspector = {'loss_total': [], 'loss_idt': [], 'loss_ssim': [], 'loss_grad': [], 'step': []}
    sample_num = len(inspector['step']) if step_num is None else step_num
    sample_step = max(len(inspector['step']) // sample_num, 1)
    x = inspector['step'][::sample_step]

    loss_total = inspector['loss_total'][::sample_step]
    loss_idt = inspector['loss_idt'][::sample_step]
    loss_grad = inspector['loss_grad'][::sample_step]

    if len(inspector['lr']) > 0:  # train inspector
        fig_n = fig_n + 1
        learning_rate = inspector['lr'][::sample_step]

    if len(x) > len(loss_total):
        x = x[:len(loss_total)]

    f = plt.figure(figsize=(40.0, 45.0))

    ax1 = f.add_subplot(fig_n, 1, 1)
    ax1.plot(x, loss_total, '.-')
    ax1.set_ylabel('loss-total')

    ax2 = f.add_subplot(fig_n, 1, 2)
    ax2.plot(x, loss_idt, '.-')
    ax2.set_ylabel('loss-idt')

    ax3 = f.add_subplot(fig_n, 1, 3)
    ax3.plot(x, loss_grad, '.-')
    ax3.set_ylabel('loss-grad')

    if len(inspector['lr']) > 0:
        ax12 = f.add_subplot(fig_n, 1, fig_n)
        ax12.plot(x, learning_rate, '.-')
        ax12.set_ylabel('lr')

    plt.savefig(save_dir)
    plt.close()


def visualize_inspector_iiw(inspector, save_dir, step_num=None, mode='Basic'):
    fig_n = 1  # subplot blocks number
    # loss_inspector = {'loss_total': [], 'loss_idt': [], 'loss_ssim': [], 'loss_grad': [], 'step': []}
    sample_num = len(inspector['step']) if step_num is None else step_num
    sample_step = max(len(inspector['step']) // sample_num, 1)
    x = inspector['step'][::sample_step]

    # if 'loss_whdr' in inspector.keys():  # test inspector
    loss = inspector['loss_whdr'][::sample_step]
    if len(inspector['loss_total']) > 0:  # train inspector
        fig_n = fig_n + 1
        loss_total = inspector['loss_total'][::sample_step]
    if len(inspector['loss_rs']) > 0:
        fig_n = fig_n + 3
        loss_rs = inspector['loss_rs'][::sample_step]
        loss_ss = inspector['loss_ss'][::sample_step]
        loss_iiw = inspector['loss_iiw'][::sample_step]
    if len(inspector['pixel_penalty']) > 0:
        fig_n = fig_n + 1
        loss_pix_penalty = inspector['pixel_penalty'][::sample_step]
    if len(inspector['preserve_info']) > 0:
        fig_n = fig_n + 1
        loss_preserve = inspector['preserve_info'][::sample_step]
    if len(inspector['perceptive']) > 0:
        fig_n = fig_n + 1
        loss_perceptive = inspector['perceptive'][::sample_step]
    if len(inspector['fea_divergence']) > 0:
        fig_n = fig_n + 1
        loss_fd = inspector['fea_divergence'][::sample_step]

    if len(x) > len(loss):
        x = x[:len(loss)]

    if fig_n > 3:
        f = plt.figure(figsize=(40.0, 60.0))
    else:
        f = plt.figure(figsize=(40.0,30.0))
    ax1 = f.add_subplot(fig_n, 1, 1)
    ax1.plot(x, loss, '.-')
    ax1.set_ylabel('loss-whdr')

    if len(inspector['loss_total']) > 0:
        ax2 = f.add_subplot(fig_n, 1, 2)
        ax2.plot(x, loss_total, '.-')
        ax2.set_ylabel('loss-total')

    if len(inspector['loss_rs']) > 0:
        ax3 = f.add_subplot(fig_n, 1, 3)
        ax3.plot(x, loss_rs, '.-')
        ax3.set_ylabel('loss_rs')

        ax4 = f.add_subplot(fig_n, 1, 4)
        ax4.plot(x, loss_ss, '.-')
        ax4.set_ylabel('loss_ss')

        ax5 = f.add_subplot(fig_n, 1, 5)
        ax5.plot(x, loss_iiw, '.-')
        ax5.set_ylabel('loss_iiw')

    if len(inspector['pixel_penalty']) > 0:
        ax6 = f.add_subplot(fig_n, 1, 6)
        ax6.plot(x, loss_pix_penalty, '.-')
        ax6.set_ylabel('pix_penalty')

    if len(inspector['preserve_info']) > 0:
        ax7 = f.add_subplot(fig_n, 1, 7)
        ax7.plot(x, loss_preserve, '.-')
        ax7.set_ylabel('loss-preserve')

    if len(inspector['perceptive']) > 0:
        ax8 = f.add_subplot(fig_n, 1, 8)
        ax8.plot(x, loss_perceptive, '.-')
        ax8.set_ylabel('loss-perceptive')

    if len(inspector['fea_divergence']) > 0:
        ax9 = f.add_subplot(fig_n, 1, 9)
        ax9.plot(x, loss_fd, '.-')
        ax9.set_ylabel('loss-fd')

    plt.savefig(save_dir)
    plt.close()


def tensor2img(tensor, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), use_norm=False):
    # mean=(0.5, 0.5, 0.5), std=(1.0, 1.0, 1.0)
    # mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)
    if type(tensor) is Variable or tensor.is_cuda:
        tensor = tensor.cpu().data
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # get the top image from a batch
    to_pil = transforms.ToPILImage()
    n_channel = tensor.shape[0]
    if use_norm and n_channel == 3:
        # color image with normalization [-x, y] ==> [0, 1]
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)

    tensor[tensor < 0] = 0
    tensor[tensor > 1] = 1
    pil_img = to_pil(tensor)
    img = np.asarray(pil_img, np.uint8)
    return img


def tensor2img_fea(tensor, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), use_norm=True):
    # mean=(0.5, 0.5, 0.5), std=(1.0, 1.0, 1.0)
    # mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)
    if type(tensor) is Variable or tensor.is_cuda:
        tensor = tensor.cpu().data
    if len(tensor.shape) == 4:
        tensor = tensor[0]  # get the top image from a batch
    to_pil = transforms.ToPILImage()
    n_channel = tensor.shape[0]
    if use_norm:
        # color image with normalization [-x, y] ==> [0, 1]
        t_min = tensor.min() if tensor.min() < 0 else 0.0
        t_max = tensor.max()
        tensor = (tensor - t_min) / (t_max - t_min)

    tensor[tensor < 0] = 0
    tensor[tensor > 1] = 1
    pil_img = to_pil(tensor)
    img = np.asarray(pil_img, np.uint8)
    return img


def show_image(image, name='image', delay=0):
    cv2.imshow(name, image)

    c = cv2.waitKey(delay)
    if chr(c & 255) == 'q':
        return False     # need to exist
    else:
        return True      # keep going


def save_eval_images(visuals, dir, id, opt):
    """output arranged as tuple [I|R|S]
       visuals: the tensor bags output by network
    """
    img_i_gt = visuals['real_I']
    img_s_gt = visuals['real_S']
    img_r_gt = visuals['real_R']
    # img_rec_gt = visuals['real_rec']
    # gt_tuple = cv2.vconcat([img_i_gt[:, :, ::-1],
    #                         img_s_gt[:, :, ::-1], img_r_gt[:, :, ::-1]])

    img_i = visuals['fake_I']
    img_s = visuals['fake_S']
    img_r = visuals['fake_R']
    # pred_tuple = cv2.vconcat([img_i[:, :, ::-1], img_s[:, :, ::-1], img_r[:, :, ::-1]])
    gt_tuple = cv2.vconcat([img_s_gt[:, :, ::-1], img_r_gt[:, :, ::-1]])
    pred_tuple = cv2.vconcat([img_s[:, :, ::-1], img_r[:, :, ::-1]])
    intrinsic_tuple = cv2.hconcat([gt_tuple, pred_tuple])

    img_s_grad_gt = visuals['real_S_grad']
    img_r_grad_gt = visuals['real_R_grad']
    img_s_grad = visuals['fake_S_grad']
    img_r_grad = visuals['fake_R_grad']
    #pdb.set_trace()
    grad_s = cv2.vconcat([img_s_grad_gt[:, :, ::-1], img_s_grad[:, :, ::-1]])
    grad_r = cv2.vconcat([img_r_grad_gt[:, :, ::-1], img_r_grad[:, :, ::-1]])
    grad_tuple = cv2.hconcat([grad_r, grad_s])

    # save_image(intrinsic_tuple, os.path.join(dir, str(id) + '_intrinsic.png'))
    # save_image(grad_tuple, os.path.join(dir, str(id) + '_0grad.png'))

    save_image(img_i_gt[:, :, ::-1], os.path.join(dir, str(id) + '_diffuse.png'))
    if opt.optim.lambda_i_w > 0:
        save_image(img_i[:, :, ::-1], os.path.join(dir, str(id) + '_diffuse-rec.png'))

    save_image(img_s[:, :, ::-1], os.path.join(dir, str(id) + '_shading-pred.png'))
    save_image(img_r[:, :, ::-1], os.path.join(dir, str(id) + '_reflect-pred.png'))
    save_image(img_s_gt[:, :, ::-1], os.path.join(dir, str(id) + '_shading-real.png'))
    save_image(img_r_gt[:, :, ::-1], os.path.join(dir, str(id) + '_reflect-real.png'))


def save_eval_images_mpi(visuals, dir, id, opt):
    """output arranged as tuple [I|R|S]
       visuals: the tensor bags output by network
       v10: use gt to correct pred colors
    """
    img_i_gt = visuals['real_I']
    img_s_gt = visuals['real_S']
    img_r_gt = visuals['real_R']
    # img_rec_gt = visuals['real_rec']
    # gt_tuple = cv2.vconcat([img_i_gt[:, :, ::-1],
    #                         img_s_gt[:, :, ::-1], img_r_gt[:, :, ::-1]])

    img_i = visuals['fake_I']
    img_s = visuals['fake_S']
    img_r = visuals['fake_R']

    img_r_ = np.zeros_like(img_r)
    for m in range(0,3):
        numerator = np.sum(np.multiply(img_r[:,:,m], img_r_gt[:,:,m]))
        denominator = np.sum(np.multiply(img_r[:,:,m], img_r[:,:,m]))
        alpha = numerator / denominator
        img_r_[:,:,m] = img_r[:,:,m] * alpha

    img_s_ = np.zeros_like(img_s)
    for m in range(0,3):
        numerator = np.sum(np.multiply(img_s[:,:,m], img_s_gt[:,:,m]))
        denominator = np.sum(np.multiply(img_s[:,:,m], img_s[:,:,m]))
        alpha = numerator / denominator
        img_s_[:,:,m] = img_s[:,:,m] * alpha

    # pred_tuple = cv2.vconcat([img_i[:, :, ::-1], img_s[:, :, ::-1], img_r[:, :, ::-1]])
    gt_tuple = cv2.vconcat([img_s_gt[:, :, ::-1], img_r_gt[:, :, ::-1]])
    pred_tuple = cv2.vconcat([img_s_[:, :, ::-1], img_r_[:, :, ::-1]])
    intrinsic_tuple = cv2.hconcat([gt_tuple, pred_tuple])

    img_s_grad_gt = visuals['real_S_grad']
    img_r_grad_gt = visuals['real_R_grad']
    img_s_grad = visuals['fake_S_grad']
    img_r_grad = visuals['fake_R_grad']
    #pdb.set_trace()
    grad_s = cv2.vconcat([img_s_grad_gt[:, :, ::-1], img_s_grad[:, :, ::-1]])
    grad_r = cv2.vconcat([img_r_grad_gt[:, :, ::-1], img_r_grad[:, :, ::-1]])
    grad_tuple = cv2.hconcat([grad_r, grad_s])

    # save_image(intrinsic_tuple, os.path.join(dir, str(id) + '_intrinsic.png'))
    save_image(grad_tuple, os.path.join(dir, str(id) + '_grad.png'))
    save_image(img_i_gt[:, :, ::-1], os.path.join(dir, str(id) + '_input.png'))
    save_image(img_s_[:, :, ::-1], os.path.join(dir, str(id) + '_shading-pred.png'))
    save_image(img_r_[:, :, ::-1], os.path.join(dir, str(id) + '_reflect-pred.png'))
    save_image(img_s_gt[:, :, ::-1], os.path.join(dir, str(id) + '_shading-real.png'))
    save_image(img_r_gt[:, :, ::-1], os.path.join(dir, str(id) + '_reflect-real.png'))
    if opt.optim.lambda_w > 0:
        save_image(img_i[:, :, ::-1], os.path.join(dir, str(id) + '_input-rec.png'))


def save_eval_images_iiw(visuals, dir, id, opt):
    """output arranged as tuple [I|R|S]
       visuals: the tensor bags output by network
    """
    img_i_gt = visuals['real_I']

    img_s_rec = visuals['rec_S']
    img_s = visuals['fake_S']
    img_r = visuals['fake_R']

    save_image(img_i_gt[:, :, ::-1], os.path.join(dir, id + '_input.png'))
    save_image(img_s[:, :], os.path.join(dir, id + '_shading-pred.png'))
    save_image(img_s_rec[:, :], os.path.join(dir, id + '_shading-rec.png'))
    save_image(img_r[:, :, ::-1], os.path.join(dir, id + '_reflect-pred.png'))


def save_image(image, dir='../imgs/0.png'):
    cv2.imwrite(dir, image)


def check_dir(s_dir, force_clean=False):
    if force_clean:
        remove_dir(s_dir)
    if not os.path.exists(s_dir):
        os.makedirs(s_dir)


def remove_dir(s_dir):
    if os.path.exists(s_dir):
        shutil.rmtree(s_dir)


def get_data_loaders(opt):
    batch_size = opt.batch_size
    num_workers = opt.num_workers

    if opt.name == 'rr_removal':
        from data import DatasetRR
        train_set = DatasetRR(data_opt=opt, is_train=True)
        test_set = DatasetRR(data_opt=opt, is_train=False)
    elif opt.name == 'toy':
        from data import DatasetToy
        train_set = DatasetToy(data_opt=opt, is_train=True)
        test_set = DatasetToy(data_opt=opt, is_train=False)
    else:
        raise NotImplementedError

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.n_iter) / float(opt.n_iter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.n_iter_decay, gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def print_network_info(net, print_struct=True):
    print('Network %s structure: ' % type(net).__name__)
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    if print_struct:
        print(net)
    print('===> In network %s, total trainable number of parameters: %d' % (type(net).__name__, num_params))


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))

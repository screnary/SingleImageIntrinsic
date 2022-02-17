from collections import OrderedDict

import utils
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import glob
from torch.autograd import grad as ta_grad
from networks_mpi_RD_v6 import get_generator, get_discriminator, GANLoss, DivergenceLoss, PerspectiveLoss
from networks_mpi_RD_v6 import Grad_Img_v1 as Grad_Img
import pytorch_ssim
import copy
import pdb

get_gradient = Grad_Img().cuda() # output 1 channel gradients
# noinspection PyAttributeOutsideInit
"""
v8: use new perspective loss: cosine + L2?; for gradient loss, use mse to
prevent bad pixels
v11: change perceptual dict the same as fd, use cosine for preserve_info
RD: refined data
RD_v6: TwoWayMS
"""

class Trainer_Basic(nn.Module):
    @staticmethod
    def name():
        return 'IID_Trainer'

    def __init__(self, t_opt):
        super(Trainer_Basic, self).__init__()
        self.opt = t_opt
        self.is_train = t_opt.is_train
        self.save_dir = t_opt.output_root

        self.weights = copy.deepcopy(t_opt.optim)
        nb = t_opt.data.batch_size
        size = t_opt.data.new_size

        self.Tensor = torch.cuda.FloatTensor if self.opt.gpu_ids else torch.Tensor
        self.input_i = None
        self.input_s = None
        self.input_r = None

        self.lr = None
        # Init the networks
        print('Constructing Networks ...')
        self.gen_split = get_generator(t_opt.model.gen, t_opt.train.mode).cuda()  # decomposition

        print('Loading Networks\' Parameters ...')
        if t_opt.continue_train:
            which_epoch = t_opt.which_epoch
            self.resume(self.gen_split, 'G_decompose', which_epoch)

        # define loss functions---need modify
        self.criterion_idt = torch.nn.L1Loss()  # L1 loss is smooth; MSELoss
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_ssim = pytorch_ssim.SSIM(window_size=11).cuda()
        self.criterion_fd = DivergenceLoss(
                        detail_weights=t_opt.model.gen.div_detail_dict,
                        cos_w=t_opt.model.gen.fd_cosw,
                        norm_w=t_opt.model.gen.fd_normw).cuda()
        self.criterion_perspective = PerspectiveLoss(
                        detail_weights=t_opt.model.gen.div_detail_dict_equal,
                        cos_w=t_opt.model.gen.p_cosw,
                        norm_w=t_opt.model.gen.p_normw).cuda()
        self.criterion_fea_extract = PerspectiveLoss(
                        detail_weights=t_opt.model.gen.div_detail_dict_equal,
                        cos_w=t_opt.model.gen.p_cosw,
                        norm_w=t_opt.model.gen.p_normw).cuda()
        self.criterion_cos_sim = torch.nn.CosineSimilarity(dim=1,
                                                           eps=1e-11).cuda()
        # self.criterionTaskAware = networks.ReflectionRemovalLoss(device_name='cuda:' + str(opt.gpu_ids[0]))
        # self.criterionTaskAware = layer_distrib_loss.DistribLoss(opt=opt, name=opt.datasetName)
        # initialize optimizers
        self.optimizer_gen = torch.optim.Adam([p for p in self.gen_split.parameters() if p.requires_grad],
                                              lr=t_opt.optim.lr_g, betas=(t_opt.optim.beta1, t_opt.optim.beta2))
        self.optimizers = []
        self.schedulers = []
        self.optimizers.append(self.optimizer_gen)
        for optimizer in self.optimizers:
            self.schedulers.append(utils.get_scheduler(optimizer, t_opt.optim))

        self.gen_split.train()

        print('---------- Networks initialized -------------')
        # utils.print_network_info(self.gen_split)
        print('---------------------------------------------')
        utils.print_network_info(self, print_struct=False)
        # pdb.set_trace()

    def forward(self):
        # fake_s, fake_r, fea_dvalue = self.gen_split(self.real_i)[:3]
        fake_s, fake_r = self.gen_split(self.real_i)[:2]
        self.fake_s = fake_s.repeat(1, 3, 1, 1)
        self.fake_r = fake_r
        self.fd_loss = None
        fake_i = fake_r * fake_s
        fake_s_grad, fake_s_gradx, fake_s_grady = get_gradient(self.fake_s)
        fake_r_grad, fake_r_gradx, fake_r_grady = get_gradient(self.fake_r)
        self.fake_i = fake_i
        self.fake_grad_s = fake_s_grad
        self.fake_s_gradx = fake_s_gradx
        self.fake_s_grady = fake_s_grady
        self.fake_grad_r = fake_r_grad
        self.fake_r_gradx = fake_r_gradx
        self.fake_r_grady = fake_r_grady
        # self.fea_dvalue = fea_dvalue  # feature divergence values [low, mid, deep, out]

    def set_input(self, input_data):
        input_i = input_data['I']
        input_s = input_data['B']
        input_r = input_data['R']

        self.img_name = input_data['name']

        # input image
        self.input_i = input_i.cuda()
        self.input_s = input_s.cuda()
        self.input_r = input_r.cuda()
        self.input_rec = self.reconstruct(self.input_s, self.input_r)

        self.real_i = Variable(self.input_i)
        self.real_s = Variable(self.input_s)
        self.real_r = Variable(self.input_r)
        self.real_rec = Variable(self.input_rec)

        # image gradients
        s_grad, s_gradx, s_grady = get_gradient(self.real_s)
        r_grad, r_gradx, r_grady = get_gradient(self.real_r)

        # pdb.set_trace()
        # [wzj-v6, grad threshold]
        valid_thre_s = s_grad.abs().mean() * 0.55
        valid_thre_r = r_grad.abs().mean() * 0.55
        s_grad[s_grad.abs() < valid_thre_s] = 0.0
        s_gradx[s_gradx.abs() < valid_thre_s] = 0.0
        s_grady[s_grady.abs() < valid_thre_s] = 0.0
        r_grad[r_grad.abs() < valid_thre_r] = 0.0
        r_gradx[r_gradx.abs() < valid_thre_r] = 0.0
        r_grady[s_grady.abs() < valid_thre_r] = 0.0

        # pdb.set_trace()
        self.real_s_gradx = s_gradx
        self.real_s_grady = s_grady
        self.real_r_gradx = r_gradx
        self.real_r_grady = r_grady
        self.real_s_grad = s_grad
        self.real_r_grad = r_grad

    def set_requires_grad(self, nets, requires_grad=False):
        """set requires_grad=False to avoid computation"""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def inference(self, input_img=None):
        self.gen_split.eval()

        with torch.no_grad():
            # reduce memory usage and speed up
            self.forward()
            self.loss_basic_computation()

            self.loss_gen_total = self.loss_gen_basic

        self.gen_split.train()

    @staticmethod
    def reconstruct(img_r, img_s, img_h=None):
        if img_h is not None:
            return img_r * img_s + img_h
        return img_r * img_s

    def gen_update(self):
        self.optimizer_gen.zero_grad()

        # compute loss
        self.loss_basic_computation()
        self.loss_gen_total = self.loss_gen_basic
        self.loss_gen_total.backward()
        # optimize
        self.optimizer_gen.step()

    def loss_basic_computation(self):
        """ compute all the loss """
        weight = self.opt.optim  # weight for optim settings
        if weight.preserve_info_w > 0:
            #diff_avg_r = self.criterion_mse(torch.mean(self.fake_r, dim=[2,3], keepdim=True),
            #                                torch.mean(self.real_r, dim=[2,3], keepdim=True))
            #diff_avg_s = self.criterion_mse(torch.mean(self.fake_s, dim=[2,3], keepdim=True),
            #                                torch.mean(self.real_s, dim=[2,3], keepdim=True))
            diff_avg_r = self.Tensor([0.0])
            diff_avg_s = self.Tensor([0.0])
            cos_diff_r = 1.0 - self.criterion_cos_sim(self.fake_r, self.real_r)  # [0,2]
            cos_diff_s = 1.0 - self.criterion_cos_sim(self.fake_s, self.real_s)
            cos_diff_r = torch.mean(cos_diff_r)
            cos_diff_s = torch.mean(cos_diff_s)
            self.loss_preserve_info = ((diff_avg_r + cos_diff_r) * weight.lambda_r_w +\
                                       (diff_avg_s + cos_diff_s) * weight.lambda_b_w) *\
                                       weight.preserve_info_w
        else:
            self.loss_preserve_info = self.Tensor([0.0])

        if weight.identity_w > 0:
            self.loss_idt_i = self.criterion_idt(self.fake_i, self.real_i) * weight.lambda_i_w
            self.loss_idt_s = self.criterion_idt(self.fake_s, self.real_s) * weight.lambda_b_w
            self.loss_idt_r = self.criterion_idt(self.fake_r, self.real_r) * weight.lambda_r_w
        else:
            self.loss_idt_i = self.Tensor([0.0])
            self.loss_idt_s = self.Tensor([0.0])
            self.loss_idt_r = self.Tensor([0.0])

        if weight.ssim_w > 0:
            self.loss_ssim_i = (1 - self.criterion_ssim(self.fake_i, self.real_i)) * weight.lambda_i_w
            self.loss_ssim_s = (1 - self.criterion_ssim(self.fake_s, self.real_s)) * weight.lambda_b_w
            self.loss_ssim_r = (1 - self.criterion_ssim(self.fake_r, self.real_r)) * weight.lambda_r_w
        else:
            self.loss_ssim_i = self.Tensor([0.0])
            self.loss_ssim_s = self.Tensor([0.0])
            self.loss_ssim_r = self.Tensor([0.0])

        if weight.gradient_w > 0:
            self.loss_grad_s = 0.15 * (self.criterion_mse(self.fake_s_gradx,
                                                         self.real_s_gradx)
                                     +self.criterion_mse(self.fake_s_grady,
                                                         self.real_s_grady)) * weight.lambda_b_w
            self.loss_grad_r = 0.15 * (self.criterion_mse(self.fake_r_gradx,
                                                         self.real_r_gradx)
                                     +self.criterion_mse(self.fake_r_grady,
                                                         self.real_r_grady)) * weight.lambda_r_w
            self.loss_grad_s += 0.85 * (self.criterion_idt(self.fake_s_gradx,
                                                    self.real_s_gradx)
                                +self.criterion_idt(self.fake_s_grady,
                                                    self.real_s_grady)) * weight.lambda_b_w
            self.loss_grad_r += 0.85 * (self.criterion_idt(self.fake_r_gradx,
                                                    self.real_r_gradx)
                                +self.criterion_idt(self.fake_r_grady,
                                                    self.real_r_grady)) * weight.lambda_r_w

            #self.loss_grad_s = (self.criterion_idt(fake_s_gradx, self.real_s_gradx) +
            #                    self.criterion_idt(fake_s_grady, self.real_s_grady)) * weight.lambda_b_w
            #self.loss_grad_r = (self.criterion_idt(fake_r_gradx, self.real_r_gradx) +
            #                    self.criterion_idt(fake_r_grady, self.real_r_grady)) * weight.lambda_r_w
        else:
            self.loss_grad_s = self.Tensor([0.0])
            self.loss_grad_r = self.Tensor([0.0])


        self.loss_feature_divergence = self.Tensor([0.0])

        self.loss_perspective_s = self.Tensor([0.0])
        self.loss_perspective_r = self.Tensor([0.0])
        self.loss_perspective = self.Tensor([0.0])

        self.loss_extract_s = self.Tensor([0.0])
        self.loss_extract_r = self.Tensor([0.0])
        self.loss_fea_extract = self.Tensor([0.0])

        self.loss_gen_idt = self.loss_idt_i + self.loss_idt_s + self.loss_idt_r
        self.loss_gen_ssim = self.loss_ssim_i + self.loss_ssim_s + self.loss_ssim_r
        self.loss_gen_grad = self.loss_grad_s + self.loss_grad_r
        # pdb.set_trace()
        # loss_gen_idt = loss_idt_s + loss_idt_r
        self.loss_gen_basic = weight.identity_w * self.loss_gen_idt + \
                              weight.ssim_w * self.loss_gen_ssim + \
                              weight.gradient_w * self.loss_gen_grad + \
                              weight.divergence_w * self.loss_feature_divergence + \
                              weight.perspective_w * self.loss_perspective + \
                              weight.fea_extract_w * self.loss_fea_extract

    def _compute_perspective_loss(self):
        fea_fake_r = self.gen_split.encoder(self.fake_r)
        fea_real_r = self.gen_split.encoder(self.real_r)
        fea_fake_s = self.gen_split.encoder(self.fake_s)
        fea_real_s = self.gen_split.encoder(self.real_s)
        self.loss_perspective_r = self.criterion_perspective(fea_fake_r,
                                                             fea_real_r,
                                                             detail_weights=None)
        self.loss_perspective_s = self.criterion_perspective(fea_fake_s,
                                                             fea_real_s,
                                                             detail_weights=None)

    def optimize_parameters(self):
        # forward
        for _ in range(1):
            self.forward()
            self.gen_update()

    def get_current_errors(self):
        """plain prediction loss"""
        ret_errors = OrderedDict()
        ret_errors['loss_total'] = self.loss_gen_total
        ret_errors['idt_I'] = self.loss_idt_i
        ret_errors['idt_S'] = self.loss_idt_s
        ret_errors['idt_R'] = self.loss_idt_r
        ret_errors['ssim_I'] = self.loss_ssim_i
        ret_errors['ssim_S'] = self.loss_ssim_s
        ret_errors['ssim_R'] = self.loss_ssim_r
        ret_errors['grad_S'] = self.loss_grad_s
        ret_errors['grad_R'] = self.loss_grad_r
        ret_errors['fea_divergence'] = self.loss_feature_divergence
        ret_errors['perspective'] = self.loss_perspective
        ret_errors['fea_extract'] = self.loss_fea_extract
        ret_errors['preserve_info'] = self.loss_preserve_info

        ret_errors['img_name'] = self.img_name

        return ret_errors

    def get_current_visuals(self):
        mean = self.opt.data.image_mean
        std = self.opt.data.image_std
        use_norm = self.opt.data.use_norm

        pred_r = self.fake_r.detach().clone()
        pred_s = self.fake_r.detach().clone()

        for b in range(self.fake_r.size(0)):
            for m in range(1,3):
                numerator = torch.dot(self.fake_r[b,m,:,:].view(-1),
                                      self.real_r[b,m,:,:].view(-1))
                denominator = torch.dot(self.fake_r[b,m,:,:].view(-1),
                                        self.fake_r[b,m,:,:].view(-1))
                alpha = numerator / denominator
                pred_r[b,m,:,:] = self.fake_r[b,m,:,:] * alpha

        for b in range(self.fake_s.size(0)):
            for m in range(1,3):
                numerator = torch.dot(self.fake_s[b,m,:,:].view(-1),
                                      self.real_s[b,m,:,:].view(-1))
                denominator = torch.dot(self.fake_s[b,m,:,:].view(-1),
                                        self.fake_s[b,m,:,:].view(-1))
                alpha = numerator / denominator
                pred_s[b,m,:,:] = self.fake_s[b,m,:,:] * alpha

        img_real_s = utils.tensor2img(self.input_s.detach().clone(), mean, std, use_norm)
        img_real_i = utils.tensor2img(self.input_i.detach().clone(), mean, std, use_norm)
        img_real_r = utils.tensor2img(self.input_r.detach().clone(), mean, std, use_norm)
        img_real_rec = utils.tensor2img(self.input_rec.detach().clone(), mean, std, use_norm)
        img_real_s_grad = utils.tensor2img(self.real_s_grad.detach().clone(), mean, std, use_norm=False)
        img_real_r_grad = utils.tensor2img(self.real_r_grad.detach().clone(), mean, std, use_norm=False)

        img_fake_s = utils.tensor2img(self.fake_s.detach().clone(), mean, std, use_norm)
        #img_fake_s = utils.tensor2img(pred_s, mean, std, use_norm)
        img_fake_r = utils.tensor2img(self.fake_r.detach().clone(), mean, std, use_norm)
        #img_fake_r = utils.tensor2img(pred_r, mean, std, use_norm)
        img_fake_i = utils.tensor2img(self.fake_i.detach().clone(), mean, std, use_norm)
        img_fake_s_grad = utils.tensor2img(self.fake_grad_s.detach().clone(), mean, std, use_norm=False)
        img_fake_r_grad = utils.tensor2img(self.fake_grad_r.detach().clone(), mean, std, use_norm=False)

        ret_visuals = OrderedDict([('real_I', img_real_rec),
                                   ('real_S', img_real_s),
                                   ('real_R', img_real_r),
                                   ('real_rec', img_real_rec),
                                   ('fake_I', img_fake_i),
                                   ('fake_S', img_fake_s),
                                   ('fake_R', img_fake_r),
                                   ('fake_S_grad', img_fake_s_grad),
                                   ('fake_R_grad', img_fake_r_grad),
                                   ('real_S_grad', img_real_s_grad),
                                   ('real_R_grad', img_real_r_grad)])

        return ret_visuals

    @staticmethod
    def loss_log(losses):
        log_detail = '\
                \t{}:{:.5f}, {}:{:.5f}, {}:{:.5f}\n \
                \t{}:{:.5f}, {}:{:.5f}, {}:{:.5f}\n \
                      \t{}:{}, {}:{}\n \
                      \t{}:{}\n \
                      \t{}:{}\n \
                      \t{}:{}\n \
                      \t{}:{}\n \
                      \t{}'.format(
                                   'loss_Shading', losses['idt_S'],
                                   'loss_Reflect', losses['idt_R'],
                                   'loss_I', losses['idt_I'],
                                   'loss_SSIM_S', losses['ssim_S'],
                                   'loss_SSIM_R', losses['ssim_R'],
                                   'loss_SSIM_I', losses['ssim_I'],
                                   'loss_grad_S', losses['grad_S'],
                                   'loss_grad_R', losses['grad_R'],
                                   'loss_fea_divergence', losses['fea_divergence'],
                                   'loss_perspective', losses['perspective'],
                                   'loss_preserve_info', losses['preserve_info'],
                                   'loss_fea_extract', losses['fea_extract'],
                                   losses['img_name'])
        return log_detail

    def resume(self, model, net_name, epoch_name):
        """resume or load model"""
        if epoch_name == 'latest' or epoch_name is None:
            model_files = glob.glob(self.save_dir + "/*.pth")
            if len(model_files):
                save_path = max(model_files)
            else:
                save_path = 'NotExist'
        else:
            save_filename = '%04d_net_%s.pth' % (epoch_name, net_name)
            save_path = os.path.join(self.save_dir, save_filename)

        # pdb.set_trace()
        if os.path.exists(save_path):
            if len(self.opt.gpu_ids) > 1:
                model = nn.DataParallel(model, devices_ids=self.opt.gpu_ids)
            model.load_state_dict(torch.load(save_path))
            print('Loding model from : %s .' % save_path)
        else:
            print('Begin a new train')
        pass

    def save_network(self, model, net_name, epoch_name):
        save_filename = '%04d_net_%s.pth' % (epoch_name, net_name)
        utils.check_dir(self.save_dir)

        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(model.cpu().state_dict(), save_path)

        model.cuda()
        if len(self.opt.gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=self.opt.gpu_ids)

    def save(self, label):
        self.save_network(self.gen_split, 'G_decompose', label)

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
        if lr <= 1.4e-6 and self.opt.use_wave_lr:  # reset lr
            #for group in self.optimizers[0].param_groups:
            #    group['lr'] = lr * 1e2
            self.refresh_optimizers(lr * 1e2)
            lr = self.optimizers[0].param_groups[0]['lr']
            print('new learning rate = %.7f' % lr)
        self.lr = lr

    def get_lr(self):
        return self.lr

    def refresh_optimizers(self, lr):
        self.optimizer_gen = torch.optim.Adam([p for p in self.gen_split.parameters() if p.requires_grad],
                                              lr=lr,
                                              betas=(self.opt.optim.beta1,
                                                     self.opt.optim.beta2))
        self.optimizers = []
        self.schedulers = []
        self.optimizers.append(self.optimizer_gen)
        for optimizer in self.optimizers:
            self.schedulers.append(utils.get_scheduler(optimizer, self.opt.optim))

    def update_optim_weights(self, ep):
        weight_old = self.weights
        if ep < 3 and self.opt.optim.ssim_w > 0 and not self.opt.continue_train:
            self.opt.optim.identity_w = weight_old.identity_w
            self.opt.optim.ssim_w = weight_old.identity_w * 0.4
        else:
            self.opt.optim.identity_w = weight_old.identity_w
            self.opt.optim.ssim_w = weight_old.ssim_w
        print('optims: idt_w {}, ssim_w {}'.format(
                                            self.opt.optim.identity_w,
                                            self.opt.optim.ssim_w))


class Trainer_GAN(nn.Module):
    @staticmethod
    def name():
        return 'IID_Trainer'

    def __init__(self, t_opt):
        super(Trainer_GAN, self).__init__()
        self.opt = t_opt
        self.is_train = t_opt.is_train
        self.save_dir = t_opt.output_root
        self.GAN_G_loop = 4
        self.GAN_D_loop = 1

        nb = t_opt.data.batch_size
        size = t_opt.data.new_size

        self.Tensor = torch.cuda.FloatTensor if self.opt.gpu_ids else torch.Tensor
        self.input_i = None
        self.input_s = None
        self.input_r = None

        # Init the networks
        print('Constructing Networks ...')
        self.gen_split = get_generator(t_opt.model.gen, t_opt.train.mode).cuda()  # decomposition
        self.dis_S = get_discriminator(t_opt.model.dis).cuda()
        self.dis_R = get_discriminator(t_opt.model.dis).cuda()

        print('Loading Networks\' Parameters ...')
        if t_opt.continue_train:
            which_epoch = t_opt.which_epoch
            self.resume(self.gen_split, 'G_decompose', which_epoch)
            self.resume(self.dis_S, 'D_S', which_epoch)
            self.resume(self.dis_R, 'D_R', which_epoch)

        # define loss functions---need modify
        self.criterion_idt = torch.nn.L1Loss()  # L1 loss is smooth; MSELoss
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_ssim = pytorch_ssim.SSIM(window_size=11).cuda()
        self.criterion_fd = DivergenceLoss().cuda()
        self.criterion_perspective = PerspectiveLoss().cuda()
        self.criterion_fea_extract = PerspectiveLoss().cuda()
        self.criterion_GAN = GANLoss(use_lsgan=True).cuda()
        # self.criterionTaskAware = networks.ReflectionRemovalLoss(device_name='cuda:' + str(opt.gpu_ids[0]))
        # self.criterionTaskAware = layer_distrib_loss.DistribLoss(opt=opt, name=opt.datasetName)
        # initialize optimizers
        self.optimizer_gen = torch.optim.Adam([p for p in self.gen_split.parameters() if p.requires_grad],
                                              lr=t_opt.optim.lr_g, betas=(t_opt.optim.beta1, t_opt.optim.beta2))
        self.optimizer_dis = torch.optim.Adam([p for p in self.dis_S.parameters() if p.requires_grad] +
                                              [p for p in self.dis_R.parameters() if p.requires_grad],
                                              lr=t_opt.optim.lr_d, betas=(t_opt.optim.beta1, t_opt.optim.beta2))
        self.optimizers = []
        self.schedulers = []
        self.optimizers.append(self.optimizer_gen)
        self.optimizers.append(self.optimizer_dis)
        for optimizer in self.optimizers:
            self.schedulers.append(utils.get_scheduler(optimizer, t_opt.optim))

        self.gen_split.train()
        self.dis_S.train()
        self.dis_R.train()

        print('---------- Networks initialized -------------')
        # utils.print_network_info(self.gen_split)
        print('---------------------------------------------')
        utils.print_network_info(self, print_struct=False)
        # pdb.set_trace()

    def forward(self):
        fake_s, fake_r, fea_dvalue = self.gen_split(self.real_i)[:3]
        self.fake_s = fake_s
        self.fake_r = fake_r
        self.fea_dvalue = fea_dvalue  # feature divergence values [low, mid, deep, out]

    def set_input(self, input_data):
        input_i = input_data['I']
        input_s = input_data['B']
        input_r = input_data['R']

        self.img_name = input_data['name']

        # input image
        self.input_i = input_i.cuda()
        self.input_s = input_s.cuda()
        self.input_r = input_r.cuda()
        self.input_rec = self.reconstruct(self.input_s, self.input_r)

        self.real_i = Variable(self.input_i)
        self.real_s = Variable(self.input_s)
        self.real_r = Variable(self.input_r)
        self.real_rec = Variable(self.input_rec)

        # image gradients
        s_grad, s_gradx, s_grady = get_gradient(self.real_s)
        r_grad, r_gradx, r_grady = get_gradient(self.real_r)
        self.real_s_gradx = s_gradx
        self.real_s_grady = s_grady
        self.real_r_gradx = r_gradx
        self.real_r_grady = r_grady
        self.real_s_grad = s_grad
        self.real_r_grad = r_grad

    def set_requires_grad(self, nets, requires_grad=False):
        """set requires_grad=False to avoid computation"""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def inference(self, input_img=None):
        self.gen_split.eval()
        self.dis_R.eval()
        self.dis_S.eval()

        weight = self.opt.optim  # weight for optim settings
        # with torch.no_grad():
            # reduce memory usage and speed up
        self.forward()
        self.loss_basic_computation()
        self.loss_gen_GAN()
        # self.loss_dis_GAN_S()
        # self.loss_dis_GAN_R()
        self.loss_dis_r = self.dis_R.calc_dis_loss(self.fake_r, self.real_r)
        self.loss_dis_s = self.dis_S.calc_dis_loss(self.fake_s, self.real_s)

        self.loss_gen_total = self.loss_gen_basic + self.loss_gen_gan * weight.gan_w

        self.gen_split.train()
        self.dis_R.train()
        self.dis_S.train()

    @staticmethod
    def reconstruct(img_r, img_s, img_h=None):
        if img_h is not None:
            return img_r * img_s + img_h
        return img_r * img_s

    def gen_update(self):
        weight = self.opt.optim  # weight for optim settings
        self.optimizer_gen.zero_grad()

        # compute loss
        self.loss_basic_computation()
        self.loss_gen_GAN()
        self.loss_gen_total = self.loss_gen_basic + self.loss_gen_gan * weight.gan_w
        self.loss_gen_total.backward()
        # optimize
        self.optimizer_gen.step()

    def dis_update(self):
        self.optimizer_dis.zero_grad()

        self.loss_dis_r = self.dis_R.calc_dis_loss(self.fake_r, self.real_r)
        self.loss_dis_s = self.dis_S.calc_dis_loss(self.fake_s, self.real_s)

        self.loss_dis_r.backward(retain_graph=True)
        self.loss_dis_s.backward(retain_graph=True)
        # optimize
        self.optimizer_dis.step()

    def dis_update_v0(self):
        self.optimizer_dis.zero_grad()

        self.loss_dis_GAN_S()
        self.loss_dis_GAN_R()
        # pdb.set_trace()  # check loss shape
        self.loss_dis_total = self.loss_dis_s + self.loss_dis_r
        self.loss_dis_total.backward(retain_graph=True)
        # optimize
        self.optimizer_dis.step()

    def loss_gen_GAN(self):
        weight = self.opt.optim  # weight for optim settings
        if weight.gan_w > 0:
            loss_G_s = self.criterion_GAN(self.dis_S(self.fake_s), True) * weight.lambda_b_w
            loss_G_r = self.criterion_GAN(self.dis_R(self.fake_r), True) * weight.lambda_r_w
        else:
            loss_G_s = Variable(self.Tensor(0.0))
            loss_G_r = Variable(self.Tensor(0.0))
        self.loss_gen_gan = loss_G_s + loss_G_r

    def loss_dis_GAN(self, netD, fake, real):
        weight = self.opt.optim  # weight for optim settings
        if weight.gan_w > 0:
            # Real
            pred_real = netD(real)
            loss_D_real = self.criterion_GAN(pred_real, True)
            # Fake
            pred_fake = netD(fake.detach())
            loss_D_fake = self.criterion_GAN(pred_fake, False)
            # Combined loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        else:
            loss_D = Variable(self.Tensor(0.0))

        grad_loss = 0.0
        if self.opt.model.dis.use_grad:
            eps = Variable(torch.rand(1), requires_grad=True)
            eps = eps.expand(real.size())
            eps = eps.cuda()
            x_tilde = eps * real + (1 - eps) * fake
            x_tilde = x_tilde.cuda()
            pred_tilde = self.calc_gen_loss(netD, x_tilde)
            gradients = ta_grad(outputs=pred_tilde, inputs=x_tilde,
                                grad_outputs=torch.ones(pred_tilde.size()).cuda(),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
            grad_val = self.opt.model.dis.grad_w * gradients
            grad_loss = ((grad_val.norm(2, dim=1) - 1) ** 2).mean()

        loss_D += grad_loss

        return loss_D

    def calc_gen_loss(self, netD, input_fake):
        # calculate the loss to train G
        pred = netD(input_fake)
        loss = self.criterion_GAN(pred, True)
        return loss

    def loss_dis_GAN_S(self):
        weight = self.opt.optim  # weight for optim settings
        self.loss_dis_s = self.loss_dis_GAN(self.dis_S, self.fake_s, self.real_s) * weight.lambda_b_w

    def loss_dis_GAN_R(self):
        weight = self.opt.optim  # weight for optim settings
        self.loss_dis_r = self.loss_dis_GAN(self.dis_R, self.fake_r, self.real_r) * weight.lambda_r_w

    def loss_basic_computation(self):
        """ compute all the loss """
        weight = self.opt.optim  # weight for optim settings
        fake_i = self.reconstruct(self.fake_s, self.fake_r, None)
        fake_s_grad, fake_s_gradx, fake_s_grady = get_gradient(self.fake_s)
        fake_r_grad, fake_r_gradx, fake_r_grady = get_gradient(self.fake_r)
        self.fake_i = fake_i
        self.fake_grad_s = fake_s_grad
        self.fake_grad_r = fake_r_grad

        if weight.identity_w > 0:
            self.loss_idt_i = self.criterion_idt(self.fake_i, self.real_i) * weight.lambda_i_w
            self.loss_idt_s = self.criterion_idt(self.fake_s, self.real_s) * weight.lambda_b_w
            self.loss_idt_r = self.criterion_idt(self.fake_r, self.real_r) * weight.lambda_r_w
        else:
            self.loss_idt_i = self.Tensor([0.0])
            self.loss_idt_s = self.Tensor([0.0])
            self.loss_idt_r = self.Tensor([0.0])

        if weight.ssim_w > 0:
            self.loss_ssim_i = (1 - self.criterion_ssim(self.fake_i, self.real_i)) * weight.lambda_i_w
            self.loss_ssim_s = (1 - self.criterion_ssim(self.fake_s, self.real_s)) * weight.lambda_b_w
            self.loss_ssim_r = (1 - self.criterion_ssim(self.fake_r, self.real_r)) * weight.lambda_r_w
        else:
            self.loss_ssim_i = self.Tensor([0.0])
            self.loss_ssim_s = self.Tensor([0.0])
            self.loss_ssim_r = self.Tensor([0.0])

        if weight.gradient_w > 0:
            self.loss_grad_s = (self.criterion_mse(fake_s_gradx, self.real_s_gradx) +
                                self.criterion_mse(fake_s_grady, self.real_s_grady)) * weight.lambda_b_w
            self.loss_grad_r = (self.criterion_mse(fake_r_gradx, self.real_r_gradx) +
                                self.criterion_mse(fake_r_grady, self.real_r_grady)) * weight.lambda_r_w
        else:
            self.loss_grad_s = self.Tensor([0.0])
            self.loss_grad_r = self.Tensor([0.0])

        if weight.divergence_w > 0:
            self.loss_feature_divergence = self.fea_dvalue[0] * weight.divergence_detail[0] + \
                                           self.fea_dvalue[1] * weight.divergence_detail[1] + \
                                           self.fea_dvalue[2] * weight.divergence_detail[2] + \
                                           self.fea_dvalue[3] * weight.divergence_detail[3]
        else:
            self.loss_feature_divergence = self.Tensor([0.0])

        if weight.perspective_w > 0:
            self._compute_perspective_loss()
            self.loss_perspective = self.loss_perspective_s * weight.lambda_b_w + \
                self.loss_perspective_r * weight.lambda_r_w
        else:
            self.loss_perspective_s = self.Tensor([0.0])
            self.loss_perspective_r = self.Tensor([0.0])
            self.loss_perspective = self.Tensor([0.0])

        if weight.fea_extract_w > 0:
            self._compute_fea_extract_loss()
            self.loss_fea_extract = self.loss_extract_s * weight.lambda_b_w + \
                self.loss_extract_r * weight.lambda_r_w
        else:
            self.loss_extract_s = self.Tensor([0.0])
            self.loss_extract_r = self.Tensor([0.0])
            self.loss_fea_extract = self.Tensor([0.0])

        self.loss_gen_idt = self.loss_idt_i + self.loss_idt_s + self.loss_idt_r
        self.loss_gen_ssim = self.loss_ssim_i + self.loss_ssim_s + self.loss_ssim_r
        self.loss_gen_grad = self.loss_grad_s + self.loss_grad_r
        # pdb.set_trace()
        # loss_gen_idt = loss_idt_s + loss_idt_r
        self.loss_gen_basic = weight.identity_w * self.loss_gen_idt + \
                              weight.ssim_w * self.loss_gen_ssim + \
                              weight.gradient_w * self.loss_gen_grad + \
                              weight.divergence_w * self.loss_feature_divergence + \
                              weight.perspective_w * self.loss_perspective + \
                              weight.fea_extract_w * self.loss_fea_extract

    def _compute_perspective_loss(self):
        fea_fake_r = self.gen_split.encoder_b(self.fake_r)
        fea_real_r = self.gen_split.encoder_b(self.real_r)
        fea_fake_s = self.gen_split.encoder_a(self.fake_s)
        fea_real_s = self.gen_split.encoder_a(self.real_s)
        self.loss_perspective_r = self.criterion_perspective(fea_fake_r, fea_real_r)
        self.loss_perspective_s = self.criterion_perspective(fea_fake_s, fea_real_s)

    def _compute_fea_extract_loss(self):
        fea_extract_s = self.gen_split.encoder_a(self.input_i)
        fea_real_s = self.gen_split.encoder_a(self.real_s)
        fea_extract_r = self.gen_split.encoder_b(self.input_i)
        fea_real_r = self.gen_split.encoder_b(self.real_r)
        self.loss_extract_s = self.criterion_fea_extract(fea_extract_s, fea_real_s)
        self.loss_extract_r = self.criterion_fea_extract(fea_extract_r, fea_real_r)

    def optimize_parameters(self):
        # forward
        for _ in range(self.GAN_G_loop):
            self.forward()
            # Gen_R and Gen_S
            self.set_requires_grad([self.dis_R, self.dis_S], False)
            self.gen_update()
        # Dis_R and Dis_S
        for _ in range(self.GAN_D_loop):
            self.forward()
            self.set_requires_grad([self.dis_R, self.dis_S], True)
            self.dis_update()

    def set_gan_gd_loop(self, G_loop=1, D_loop=1):
        self.GAN_G_loop = G_loop
        self.GAN_D_loop = D_loop

    def get_current_errors(self):
        """plain prediction loss"""
        ret_errors = OrderedDict()
        ret_errors['loss_total'] = self.loss_gen_total
        ret_errors['idt_I'] = self.loss_idt_i
        ret_errors['idt_S'] = self.loss_idt_s
        ret_errors['idt_R'] = self.loss_idt_r
        ret_errors['ssim_I'] = self.loss_ssim_i
        ret_errors['ssim_S'] = self.loss_ssim_s
        ret_errors['ssim_R'] = self.loss_ssim_r
        ret_errors['grad_S'] = self.loss_grad_s
        ret_errors['grad_R'] = self.loss_grad_r
        ret_errors['fea_divergence'] = self.loss_feature_divergence
        ret_errors['perspective'] = self.loss_perspective
        ret_errors['fea_extract'] = self.loss_fea_extract

        ret_errors['gan_G'] = self.loss_gen_gan
        ret_errors['gan_D_r'] = self.loss_dis_r
        ret_errors['gan_D_s'] = self.loss_dis_s

        ret_errors['img_name'] = self.img_name

        return ret_errors

    def get_current_visuals(self):
        mean = self.opt.data.image_mean
        std = self.opt.data.image_std
        use_norm = self.opt.data.use_norm

        img_real_s = utils.tensor2img(self.input_s.detach().clone(), mean, std, use_norm)
        img_real_i = utils.tensor2img(self.input_i.detach().clone(), mean, std, use_norm)
        img_real_r = utils.tensor2img(self.input_r.detach().clone(), mean, std, use_norm)
        img_real_rec = utils.tensor2img(self.input_rec.detach().clone(), mean, std, use_norm)
        img_real_s_grad = utils.tensor2img(self.real_s_grad.detach().clone(), mean, std, use_norm=False)
        img_real_r_grad = utils.tensor2img(self.real_r_grad.detach().clone(), mean, std, use_norm=False)

        img_fake_s = utils.tensor2img(self.fake_s.detach().clone(), mean, std, use_norm)
        img_fake_r = utils.tensor2img(self.fake_r.detach().clone(), mean, std, use_norm)
        img_fake_i = utils.tensor2img(self.fake_i.detach().clone(), mean, std, use_norm)
        img_fake_s_grad = utils.tensor2img(self.fake_grad_s.detach().clone(), mean, std, use_norm=False)
        img_fake_r_grad = utils.tensor2img(self.fake_grad_r.detach().clone(), mean, std, use_norm=False)

        ret_visuals = OrderedDict([('real_I', img_real_i),
                                   ('real_S', img_real_s),
                                   ('real_R', img_real_r),
                                   ('real_rec', img_real_rec),
                                   ('fake_I', img_fake_i),
                                   ('fake_S', img_fake_s),
                                   ('fake_R', img_fake_r),
                                   ('fake_S_grad', img_fake_s_grad),
                                   ('fake_R_grad', img_fake_r_grad),
                                   ('real_S_grad', img_real_s_grad),
                                   ('real_R_grad', img_real_r_grad)])

        return ret_visuals

    @staticmethod
    def loss_log(losses):
        log_detail = '*****************loss details\n\
                      \t{}:{}, {}:{}\n \
                      \t{}:{}, {}:{}\n \
                      \t{}:{}, {}:{}\n \
                      \t{}:{}\n \
                      \t{}:{}\n \
                      \t{}:{}\n \
                      \t{}:{}\n \
                      \t{}:{}\n \
                      \t{}:{}\n \
                      \t{}'.format(
                                   'loss_Shading', losses['idt_S'],
                                   'loss_Reflect', losses['idt_R'],
                                   'loss_SSIM_S', losses['ssim_S'],
                                   'loss_SSIM_R', losses['ssim_R'],
                                   'loss_grad_S', losses['grad_S'],
                                   'loss_grad_R', losses['grad_R'],
                                   'loss_fea_divergence', losses['fea_divergence'],
                                   'loss_perspective', losses['perspective'],
                                   'loss_fea_extract', losses['fea_extract'],
                                   'loss_GAN_G', losses['gan_G'],
                                   'loss_GAN_D_r', losses['gan_D_r'],
                                   'loss_GAN_D_s', losses['gan_D_s'],
                                   losses['img_name'])
        return log_detail

    def resume(self, model, net_name, epoch_name):
        """resume or load model"""
        if epoch_name == 'latest' or epoch_name is None:
            model_files = glob.glob(self.save_dir + "/*.pth")
            if len(model_files):
                save_path = max(model_files)
            else:
                save_path = 'NotExist'
        else:
            save_filename = '%04d_net_%s.pth' % (epoch_name, net_name)
            save_path = os.path.join(self.save_dir, save_filename)

        # pdb.set_trace()
        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path))
            print('Loding model from : %s .' % save_path)
        else:
            print('Begin a new train')

    def save_network(self, model, net_name, epoch_name):
        save_filename = '%04d_net_%s.pth' % (epoch_name, net_name)
        utils.check_dir(self.save_dir)

        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(model.cpu().state_dict(), save_path)

        model.cuda()
        if len(self.opt.gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=self.opt.gpu_ids)

    def save(self, label):
        self.save_network(self.gen_split, 'G_decompose', label)
        self.save_network(self.dis_S, 'D_S', label)
        self.save_network(self.dis_R, 'D_R', label)

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)


class Trainer_GAN_v0(nn.Module):
    @staticmethod
    def name():
        return 'IID_Trainer'

    def __init__(self, t_opt):
        super(Trainer_GAN_v0, self).__init__()
        self.opt = t_opt
        self.is_train = t_opt.is_train
        self.save_dir = t_opt.output_root

        nb = t_opt.data.batch_size
        size = t_opt.data.new_size

        self.Tensor = torch.cuda.FloatTensor if self.opt.gpu_ids else torch.Tensor
        self.input_i = None
        self.input_s = None
        self.input_r = None

        # Init the networks
        print('Constructing Networks ...')
        self.gen_split = get_generator(t_opt.model.gen, t_opt.train.mode).cuda()  # decomposition
        self.dis_S = get_discriminator(t_opt.model.dis).cuda()
        self.dis_R = get_discriminator(t_opt.model.dis).cuda()
        # self.gen_combine = self.combine  # reconstruction

        print('Loading Networks\' Parameters ...')
        which_epoch = t_opt.which_epoch
        self.resume(self.gen_split, 'G_decompose', which_epoch)

        # define loss functions---need modify
        self.criterion_idt = torch.nn.L1Loss()  # L1 loss is smooth; MSELoss
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_ssim = pytorch_ssim.SSIM(window_size=11).cuda()
        self.criterion_GAN = GANLoss(use_lsgan=True).cuda()
        # self.criterionTaskAware = networks.ReflectionRemovalLoss(device_name='cuda:' + str(opt.gpu_ids[0]))
        # self.criterionTaskAware = layer_distrib_loss.DistribLoss(opt=opt, name=opt.datasetName)
        # initialize optimizers
        self.optimizer_gen = torch.optim.Adam([p for p in self.gen_split.parameters() if p.requires_grad],
                                              lr=t_opt.optim.lr_g, betas=(t_opt.optim.beta1, t_opt.optim.beta2))
        self.optimizer_dis = torch.optim.Adam([p for p in self.dis_S.parameters() if p.requires_grad] +
                                              [p for p in self.dis_R.parameters() if p.requires_grad],
                                              lr=t_opt.optim.lr_d, betas=(t_opt.optim.beta1, t_opt.optim.beta2))
        self.optimizers = []
        self.schedulers = []
        self.optimizers.append(self.optimizer_gen)
        self.optimizers.append(self.optimizer_dis)
        for optimizer in self.optimizers:
            self.schedulers.append(utils.get_scheduler(optimizer, t_opt.optim))

        self.gen_split.train()
        self.dis_S.train()
        self.dis_R.train()

        print('---------- Networks initialized -------------')
        # utils.print_network_info(self.gen_split)
        print('---------------------------------------------')
        utils.print_network_info(self, print_struct=False)
        # pdb.set_trace()

    def forward(self):
        fake_s, fake_r = self.gen_split(self.real_i)[:2]
        self.fake_s = fake_s
        self.fake_r = fake_r

    def set_input(self, input_data):
        input_i = input_data['I']
        input_s = input_data['B']
        input_r = input_data['R']

        self.img_name = input_data['name']

        # input image
        self.input_i = input_i.cuda()
        self.input_s = input_s.cuda()
        self.input_r = input_r.cuda()
        self.input_rec = self.reconstruct(self.input_s, self.input_r)

        self.real_i = Variable(self.input_i)
        self.real_s = Variable(self.input_s)
        self.real_r = Variable(self.input_r)
        self.real_rec = Variable(self.input_rec)

        # image gradients
        s_grad, s_gradx, s_grady = get_gradient(self.real_s)
        r_grad, r_gradx, r_grady = get_gradient(self.real_r)
        self.real_s_gradx = s_gradx
        self.real_s_grady = s_grady
        self.real_r_gradx = r_gradx
        self.real_r_grady = r_grady
        self.real_s_grad = s_grad
        self.real_r_grad = r_grad

    def set_requires_grad(self, nets, requires_grad=False):
        """set requires_grad=False to avoid computation"""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def inference(self, input_img=None):
        self.gen_split.eval()
        self.dis_R.eval()
        self.dis_S.eval()

        weight = self.opt.optim  # weight for optim settings
        with torch.no_grad():
            # reduce memory usage and speed up
            self.forward()
            self.loss_basic_computation()
            self.loss_gen_GAN()
            self.loss_dis_GAN_S()
            self.loss_dis_GAN_R()

            self.loss_gen_total = self.loss_gen_basic + self.loss_gen_gan * weight.gan_w

        self.gen_split.train()
        self.dis_R.train()
        self.dis_S.train()

    @staticmethod
    def reconstruct(img_r, img_s, img_h=None):
        if img_h is not None:
            return img_r * img_s + img_h
        return img_r * img_s

    def gen_update(self):
        weight = self.opt.optim  # weight for optim settings
        self.optimizer_gen.zero_grad()

        # compute loss
        self.loss_basic_computation()
        self.loss_gen_GAN()
        self.loss_gen_total = self.loss_gen_basic + self.loss_gen_gan * weight.gan_w
        self.loss_gen_total.backward()
        # optimize
        self.optimizer_gen.step()

    def dis_update(self):
        self.optimizer_dis.zero_grad()

        self.loss_dis_GAN_S()
        self.loss_dis_GAN_R()
        # pdb.set_trace()  # check loss shape
        self.loss_dis_s.backward()
        self.loss_dis_r.backward()
        # optimize
        self.optimizer_dis.step()

    def loss_gen_GAN(self):
        weight = self.opt.optim  # weight for optim settings
        if weight.gan_w > 0:
            loss_G_s = self.criterion_GAN(self.dis_S(self.fake_s), True) * weight.lambda_b_w
            loss_G_r = self.criterion_GAN(self.dis_R(self.fake_r), True) * weight.lambda_r_w
        else:
            loss_G_s = Variable(self.Tensor(0.0))
            loss_G_r = Variable(self.Tensor(0.0))
        self.loss_gen_gan = loss_G_s + loss_G_r

    def loss_dis_GAN(self, netD, fake, real):
        weight = self.opt.optim  # weight for optim settings
        if weight.gan_w > 0:
            # Real
            pred_real = netD(real)
            loss_D_real = self.criterion_GAN(pred_real, True)
            # Fake
            pred_fake = netD(fake.detach())
            loss_D_fake = self.criterion_GAN(pred_fake, False)
            # Combined loss
            loss_D = (loss_D_real + loss_D_fake) * 0.5
        else:
            loss_D = Variable(self.Tensor(0.0))

        return loss_D

    def loss_dis_GAN_S(self):
        weight = self.opt.optim  # weight for optim settings
        self.loss_dis_s = self.loss_dis_GAN(self.dis_S, self.fake_s, self.real_s) * weight.lambda_b_w

    def loss_dis_GAN_R(self):
        weight = self.opt.optim  # weight for optim settings
        self.loss_dis_r = self.loss_dis_GAN(self.dis_R, self.fake_r, self.real_r) * weight.lambda_r_w

    def loss_basic_computation(self):
        """ compute all the loss """
        weight = self.opt.optim  # weight for optim settings
        fake_i = self.reconstruct(self.fake_s, self.fake_r, None)
        fake_s_grad, fake_s_gradx, fake_s_grady = get_gradient(self.fake_s)
        fake_r_grad, fake_r_gradx, fake_r_grady = get_gradient(self.fake_r)
        self.fake_i = fake_i
        self.fake_grad_s = fake_s_grad
        self.fake_grad_r = fake_r_grad

        if weight.identity_w > 0:
            self.loss_idt_i = self.criterion_idt(self.fake_i, self.real_i) * weight.lambda_i_w
            self.loss_idt_s = self.criterion_idt(self.fake_s, self.real_s) * weight.lambda_b_w
            self.loss_idt_r = self.criterion_idt(self.fake_r, self.real_r) * weight.lambda_r_w
        else:
            self.loss_idt_i = 0
            self.loss_idt_s = 0
            self.loss_idt_r = 0

        if weight.ssim_w > 0:
            self.loss_ssim_i = (1 - self.criterion_ssim(self.fake_i, self.real_i)) * weight.lambda_i_w
            self.loss_ssim_s = (1 - self.criterion_ssim(self.fake_s, self.real_s)) * weight.lambda_b_w
            self.loss_ssim_r = (1 - self.criterion_ssim(self.fake_r, self.real_r)) * weight.lambda_r_w
        else:
            self.loss_ssim_i = 0
            self.loss_ssim_s = 0
            self.loss_ssim_r = 0

        if weight.gradient_w > 0:
            self.loss_grad_s = (self.criterion_mse(fake_s_gradx, self.real_s_gradx) +
                                self.criterion_mse(fake_s_grady, self.real_s_grady)) * weight.lambda_b_w
            self.loss_grad_r = (self.criterion_mse(fake_r_gradx, self.real_r_gradx) +
                                self.criterion_mse(fake_r_grady, self.real_r_grady)) * weight.lambda_r_w
        else:
            self.loss_grad_s = 0
            self.loss_grad_r = 0

        self.loss_gen_idt = self.loss_idt_i + self.loss_idt_s + self.loss_idt_r
        self.loss_gen_ssim = self.loss_ssim_i + self.loss_ssim_s + self.loss_ssim_r
        self.loss_gen_grad = self.loss_grad_s + self.loss_grad_r
        # pdb.set_trace()
        # loss_gen_idt = loss_idt_s + loss_idt_r
        self.loss_gen_basic = weight.identity_w * self.loss_gen_idt + \
                              weight.ssim_w * self.loss_gen_ssim + \
                              weight.gradient_w * self.loss_gen_grad

    def optimize_parameters(self):
        # forward
        self.forward()
        # Gen_R and Gen_S
        self.set_requires_grad([self.dis_R, self.dis_S], False)
        self.gen_update()
        # Dis_R and Dis_S
        self.set_requires_grad([self.dis_R, self.dis_S], True)
        self.dis_update()

    def get_current_errors(self):
        """plain prediction loss"""
        ret_errors = OrderedDict()
        ret_errors['loss_total'] = self.loss_gen_total
        ret_errors['idt_I'] = self.loss_idt_i
        ret_errors['idt_S'] = self.loss_idt_s
        ret_errors['idt_R'] = self.loss_idt_r
        ret_errors['ssim_I'] = self.loss_ssim_i
        ret_errors['ssim_S'] = self.loss_ssim_s
        ret_errors['ssim_R'] = self.loss_ssim_r
        ret_errors['grad_S'] = self.loss_grad_s
        ret_errors['grad_R'] = self.loss_grad_r

        ret_errors['gan_G'] = self.loss_gen_gan
        ret_errors['gan_D_r'] = self.loss_dis_r
        ret_errors['gan_D_s'] = self.loss_dis_s
        ret_errors['img_name'] = self.img_name

        return ret_errors

    def get_current_visuals(self):
        mean = self.opt.data.image_mean
        std = self.opt.data.image_std
        use_norm = self.opt.data.use_norm

        img_real_s = utils.tensor2img(self.input_s, mean, std, use_norm)
        img_real_r = utils.tensor2img(self.input_r, mean, std, use_norm)
        img_real_i = utils.tensor2img(self.input_i, mean, std, use_norm)
        img_real_rec = utils.tensor2img(self.input_rec, mean, std, use_norm)
        img_real_s_grad = utils.tensor2img(self.real_s_grad, mean, std, use_norm)
        img_real_r_grad = utils.tensor2img(self.real_r_grad, mean, std, use_norm)

        img_fake_s = utils.tensor2img(self.fake_s, mean, std, use_norm)
        img_fake_r = utils.tensor2img(self.fake_r, mean, std, use_norm)
        img_fake_i = utils.tensor2img(self.fake_i, mean, std, use_norm)
        img_fake_s_grad = utils.tensor2img(self.fake_grad_s, mean, std, use_norm)
        img_fake_r_grad = utils.tensor2img(self.fake_grad_r, mean, std, use_norm)

        ret_visuals = OrderedDict([('real_I', img_real_i),
                                   ('real_S', img_real_s),
                                   ('real_R', img_real_r),
                                   ('real_rec', img_real_rec),
                                   ('fake_I', img_fake_i),
                                   ('fake_S', img_fake_s),
                                   ('fake_R', img_fake_r),
                                   ('fake_S_grad', img_fake_s_grad),
                                   ('fake_R_grad', img_fake_r_grad),
                                   ('real_S_grad', img_real_s_grad),
                                   ('real_R_grad', img_real_r_grad)])

        return ret_visuals

    @staticmethod
    def loss_log(losses):
        log_detail = '\t{}:{}, {}:{}, {}:{}, {}:{}\n \
                      \t{}:{}, {}:{}\n \
                      \t{}:{}, {}:{}\n \
                      \t{}:{}, {}:{}, {}:{}\n \
                      \t{}'.format(
                                   'loss_total', losses['loss_total'],
                                   'loss_Shading', losses['idt_S'],
                                   'loss_Reflect', losses['idt_R'],
                                   'loss_Reconst', losses['idt_I'],
                                   'loss_SSIM_S', losses['ssim_S'],
                                   'loss_SSIM_R', losses['ssim_R'],
                                   'loss_grad_S', losses['grad_S'],
                                   'loss_grad_R', losses['grad_R'],
                                   'loss_GAN_G', losses['gan_G'],
                                   'loss_GAN_D_r', losses['gan_D_r'],
                                   'loss_GAN_D_s', losses['gan_D_s'],
                                   losses['img_name'])
        return log_detail

    def resume(self, model, net_name, epoch_name):
        """resume or load model"""
        # Todo: resume discriminators
        if epoch_name == 'latest' or epoch_name is None:
            model_files = glob.glob(self.save_dir + "/*.pth")
            if len(model_files):
                save_path = max(model_files)
            else:
                save_path = 'NotExist'
        else:
            save_filename = '%04d_net_%s.pth' % (epoch_name, net_name)
            save_path = os.path.join(self.save_dir, save_filename)

        if os.path.exists(save_path):
            model.load_state_dict(torch.load(save_path))
            print('Loding model from : %s .' % save_path)
        else:
            print('Begin a new train')

    def save_network(self, model, net_name, epoch_name):
        save_filename = '%04d_net_%s.pth' % (epoch_name, net_name)
        utils.check_dir(self.save_dir)

        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(model.cpu().state_dict(), save_path)

        model.cuda()
        if len(self.opt.gpu_ids) > 1:
            model = nn.DataParallel(model, device_ids=self.opt.gpu_ids)

    def save(self, label):
        self.save_network(self.gen_split, 'G_decompose', label)
        self.save_network(self.dis_S, 'D_S', label)
        self.save_network(self.dis_R, 'D_R', label)

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

from collections import OrderedDict

import utils
from torch.autograd import Variable
import torch
import torch.nn as nn
import os
import glob
from torch.autograd import grad as ta_grad
from networks_DI import get_generator
from networks_DI import Grad_Img_v1 as Grad_Img

import copy
import pdb

get_gradient = Grad_Img().cuda()  # output 1 channel gradients
# noinspection PyAttributeOutsideInit
"""
v8: use new perspective loss: cosine + L2?; for gradient loss, use mse to
prevent bad pixels
v11: change perceptual dict the same as fd, use cosine for preserve_info
RD: refined data
DirectIntrinsics: this
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
        self.padedge = 0
        self.padimg = nn.ReplicationPad2d(self.padedge)
        self.criterion_idt = torch.nn.L1Loss()  # L1 loss is smooth; MSELoss
        self.criterion_mse = torch.nn.MSELoss()
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
        if self.padedge > 0:
            fake_s, fake_r = self.gen_split(self.padimg(self.real_i))[:2]
            self.fake_s = fake_s[:,:,self.padedge:-self.padedge,
                                 self.padedge:-self.padedge].repeat(1, 3, 1, 1)
            self.fake_r = fake_r[:,:,self.padedge:-self.padedge,
                                 self.padedge:-self.padedge]
        else:
            fake_s, fake_r = self.gen_split(self.real_i)[:2]
            self.fake_s = fake_s.repeat(1, 3, 1, 1)
            self.fake_r = fake_r

        fake_s_grad, fake_s_gradx, fake_s_grady = get_gradient(self.fake_s)
        fake_r_grad, fake_r_gradx, fake_r_grady = get_gradient(self.fake_r)
        self.fake_i = self.fake_r * self.fake_s
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

        if weight.identity_w > 0:
            self.loss_idt_i = self.criterion_idt(self.fake_i, self.real_i) * weight.lambda_i_w
            self.loss_idt_s = self.criterion_idt(self.fake_s, self.real_s) * weight.lambda_b_w
            self.loss_idt_r = self.criterion_idt(self.fake_r, self.real_r) * weight.lambda_r_w
        else:
            self.loss_idt_i = self.Tensor([0.0])
            self.loss_idt_s = self.Tensor([0.0])
            self.loss_idt_r = self.Tensor([0.0])


        if weight.gradient_w > 0:
            self.loss_grad_s = 1.0 * (self.criterion_mse(self.fake_s_gradx,
                                                         self.real_s_gradx)
                                     +self.criterion_mse(self.fake_s_grady,
                                                         self.real_s_grady)) * weight.lambda_b_w
            self.loss_grad_r = 1.0 * (self.criterion_mse(self.fake_r_gradx,
                                                         self.real_r_gradx)
                                     +self.criterion_mse(self.fake_r_grady,
                                                         self.real_r_grady)) * weight.lambda_r_w
            self.loss_grad_s += 0.0 * (self.criterion_idt(self.fake_s_gradx,
                                                    self.real_s_gradx)
                                +self.criterion_idt(self.fake_s_grady,
                                                    self.real_s_grady)) * weight.lambda_b_w
            self.loss_grad_r += 0.0 * (self.criterion_idt(self.fake_r_gradx,
                                                    self.real_r_gradx)
                                +self.criterion_idt(self.fake_r_grady,
                                                    self.real_r_grady)) * weight.lambda_r_w
        else:
            self.loss_grad_s = self.Tensor([0.0])
            self.loss_grad_r = self.Tensor([0.0])

        self.loss_gen_idt = self.loss_idt_i + self.loss_idt_s + self.loss_idt_r
        self.loss_gen_grad = self.loss_grad_s + self.loss_grad_r
        # pdb.set_trace()
        # loss_gen_idt = loss_idt_s + loss_idt_r
        self.loss_gen_basic = weight.identity_w * self.loss_gen_idt + \
                              weight.gradient_w * self.loss_gen_grad

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

        ret_errors['grad_S'] = self.loss_grad_s
        ret_errors['grad_R'] = self.loss_grad_r

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
                numerator = torch.dot(self.fake_r[b,m,:,:].contiguous().view(-1),
                                      self.real_r[b,m,:,:].contiguous().view(-1))
                denominator = torch.dot(self.fake_r[b,m,:,:].contiguous().view(-1),
                                        self.fake_r[b,m,:,:].contiguous().view(-1))
                alpha = numerator / denominator
                pred_r[b,m,:,:] = self.fake_r[b,m,:,:] * alpha

        for b in range(self.fake_s.size(0)):
            for m in range(1,3):
                numerator = torch.dot(self.fake_s[b,m,:,:].contiguous().view(-1),
                                      self.real_s[b,m,:,:].contiguous().view(-1))
                denominator = torch.dot(self.fake_s[b,m,:,:].contiguous().view(-1),
                                        self.fake_s[b,m,:,:].contiguous().view(-1))
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
                      \t{}:{}, {}:{}\n \
                      \t{}'.format(
                                   'loss_Shading', losses['idt_S'],
                                   'loss_Reflect', losses['idt_R'],
                                   'loss_I', losses['idt_I'],
                                   'loss_grad_S', losses['grad_S'],
                                   'loss_grad_R', losses['grad_R'],
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
        pass

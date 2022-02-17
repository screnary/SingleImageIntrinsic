from collections import OrderedDict

import utils
from utils.image_pool import ImagePool
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
from torch.autograd import grad as ta_grad
from networks_saw import get_generator, JointLoss, PerspectiveLoss
from networks_saw import Grad_Img_v1 as Grad_Img
import pytorch_ssim

import numpy as np
import saw_utils
import h5py
import os.path
import scipy.misc
from scipy.ndimage.filters import maximum_filter
import skimage
from scipy.ndimage.measurements import label
from skimage.transform import resize
import pdb


"""
this code is for test on SAW dataset
"""

# noinspection PyAttributeOutsideInit
get_gradient = Grad_Img().cuda()  # output 1 channel gradients

# copied from trainer_mpi_RD_v10_test.py


class Trainer_Basic(nn.Module):
    @staticmethod
    def name():
        return 'IID_Trainer_iiw'

    def __init__(self, t_opt):
        super(Trainer_Basic, self).__init__()
        self.opt = t_opt
        self.is_train = t_opt.is_train
        self.save_dir = t_opt.output_root

        self.threshold = t_opt.optim.threshold
        self.thre_decay = t_opt.optim.thre_decay
        self.albedo_min = t_opt.optim.albedo_min
        self.shading_max = t_opt.optim.shading_max

        nb = t_opt.data.batch_size
        size = t_opt.data.new_size

        self.Tensor = torch.cuda.FloatTensor if self.opt.gpu_ids else torch.Tensor
        self.input_i = None
        # self.input_s = None
        # self.input_r = None

        self.lr = None

        # Init the networks
        print('Constructing Networks ...')
        self.gen_decompose = get_generator(t_opt.model.gen, t_opt.train.mode).cuda()  # decomposition

        self.fake_R_pool = ImagePool(self.opt.train.pool_size)
        self.fake_S_pool = ImagePool(self.opt.train.pool_size)

        print('Loading Networks\' Parameters ...')
        if t_opt.continue_train:
            which_epoch = t_opt.which_epoch
            self.resume(self.gen_decompose, 'G_decompose', which_epoch)

        # define loss functions---need modify
        self.criterion_joint = JointLoss(self.opt.optim)

        self.padedge = 15
        self.padimg = nn.ReplicationPad2d(self.padedge)
        self.criterion_idt = torch.nn.L1Loss()  # [L1Loss()] L1 loss is smooth; MSELoss
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_ssim = pytorch_ssim.SSIM(window_size=11).cuda()

        self.criterion_perspective = PerspectiveLoss(
            detail_weights=t_opt.model.gen.div_detail_dict_equal,
            cos_w=t_opt.model.gen.p_cosw,
            norm_w=t_opt.model.gen.p_normw).cuda()

        # initialize optimizers
        self.optimizer_gen = torch.optim.Adam([p for p in self.gen_decompose.parameters() if p.requires_grad],
                                              lr=t_opt.optim.lr_g, betas=(t_opt.optim.beta1, t_opt.optim.beta2))
        self.optimizers = []
        self.schedulers = []
        self.optimizers.append(self.optimizer_gen)
        for optimizer in self.optimizers:
            self.schedulers.append(utils.get_scheduler(optimizer, t_opt.optim))

        self.gen_decompose.train()

        print('---------- Networks initialized -------------')
        # utils.print_network_info(self.gen_decompose)
        print('---------------------------------------------')
        utils.print_network_info(self, print_struct=False)
        # pdb.set_trace()

    def forward(self):
        if 'oneway' in self.opt.train.mode:

            fake_s, fake_r, _ = self.gen_decompose(self.padimg(self.real_i))[:3]
            self.fake_s = fake_s[:, :, self.padedge:-(self.padedge), self.padedge:-(self.padedge)]
            self.fake_r = fake_r[:, :, self.padedge:-(self.padedge), self.padedge:-(self.padedge)]

            self.rec_s = self.s_from_r()
        elif 'cross' in self.opt.train.mode:
            fake_s, fake_r, fd_loss = self.gen_decompose(self.padimg(self.real_i))[:3]
            self.fake_s = fake_s[:, :, self.padedge:-(self.padedge), self.padedge:-(self.padedge)]
            self.fake_r = fake_r[:, :, self.padedge:-(self.padedge), self.padedge:-(self.padedge)]

            self.rec_s = self.s_from_r().detach()
            self.fd_loss = fd_loss  # feature divergence value of [low, mid, mid2, deep, out]

        rec_s_grad, rec_s_gradx, rec_s_grady = get_gradient(self.rec_s.repeat(1, 3, 1, 1))
        valid_thre_s = rec_s_grad.abs().mean() * 0.45
        rec_s_grad[rec_s_grad.abs() < valid_thre_s] = 0.0
        rec_s_gradx[rec_s_grad.abs() < valid_thre_s] = 0.0
        rec_s_grady[rec_s_grad.abs() < valid_thre_s] = 0.0
        self.rec_s_grad = rec_s_grad
        self.rec_s_gradx = rec_s_gradx
        self.rec_s_grady = rec_s_grady

    def set_input(self, input_data, targets):
        self.num_pair = input_data.size(0)  # stacked iamges num
        self.input_i = input_data.cuda()
        self.targets = targets

        self.real_i = Variable(self.input_i, requires_grad=False)

    def set_requires_grad(self, nets, requires_grad=False):
        """set requires_grad=False to avoid computation"""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    @staticmethod
    def reconstruct(img_r, img_s, img_h=None):
        if img_h is not None:
            return img_r * img_s + img_h
        return img_r * img_s

    def IIWReconstLoss(self, R, S, targets):
        S = S.repeat(1, 3, 1, 1)
        rgb_img = Variable(targets['rgb_img'].cuda(), requires_grad=False)

        # 1 channel
        chromaticity = Variable(targets['chromaticity'].cuda(), requires_grad=False)
        p_R = torch.mul(chromaticity, R.repeat(1, 3, 1, 1))

        # return torch.mean( torch.mul(L, torch.pow( torch.log(rgb_img) - torch.log(p_R) - torch.log(S), 2)))
        return torch.mean(torch.pow(rgb_img - torch.mul(p_R, S), 2))

    def s_from_r(self):
        rgb_img = Variable(self.targets['rgb_img'].cuda(), requires_grad=False)  # (N,3,256,384)
        # pdb.set_trace()
        p_S = torch.div(torch.mean(rgb_img, dim=1, keepdim=True), self.fake_r+1e-12)
        p_S_reg = p_S.clone()
        if self.opt.is_debug:
            # pdb.set_trace()
            pass
        p_S_reg[p_S > 1.0] = 1.0  # regularized p_S
        return p_S_reg

    def inference(self, input_img=None):
        self.gen_decompose.eval()

        with torch.no_grad():
            # reduce memory usage and speed up
            self.forward()
            # pdb.set_trace()
            # self.loss_basic_computation()  # cannot compute in inference phase
            # self.loss_gen_total = self.loss_gen_basic

        self.gen_decompose.train()

    def gen_update(self):
        weight = self.opt.optim

        self.optimizer_gen.zero_grad()

        # compute loss
        self.loss_basic_computation()
        self.loss_cross_computation()

        self.loss_gen_total = self.loss_gen_joint + self.loss_gen_cross *\
                              weight.iiw_cross_w

        self.loss_gen_total.backward()
        # optimize
        self.optimizer_gen.step()

    def loss_basic_computation(self):
        """ compute all the loss """
        weight = self.opt.optim  # weight for optim settings
        if 'cross' in self.opt.train.mode:
            self.loss_joint, self.loss_rs, self.loss_ss, self.loss_iiw = self.criterion_joint(
                                                                         self.real_i, self.fake_r, self.rec_s,  # fake_s
                                                                         self.targets)
        elif 'oneway' in self.opt.train.mode:
            self.loss_joint, self.loss_rs, self.loss_ss, self.loss_iiw = self.criterion_joint(
                                                                         self.real_i, self.fake_r, self.rec_s,
                                                                         self.targets)
        self.loss_joint_var = self.criterion_joint.get_loss_var()

        # pdb.set_trace()
        # loss_gen_idt = loss_idt_s + loss_idt_r
        self.loss_gen_joint = self.loss_joint_var

    def loss_cross_computation(self):
        """ compute all the cross loss """
        weight = self.opt.optim  # weight for optim settings

        if weight.preserve_info_w > 0:
            # rgb_img = Variable(self.targets['rgb_img'].cuda(), requires_grad=False)
            rgb_img = self.real_i
            self.loss_preserve_info = (
                                       torch.pow(torch.mean(self.fake_r) - torch.mean(rgb_img), 2) * 0.05 +
                                       torch.pow(torch.mean(self.fake_r) - torch.mean(self.rec_s), 2) * 0.1 +
                                       torch.pow(torch.mean(self.fake_s) - torch.mean(self.rec_s), 2) * 0.6 +
                                       torch.pow(torch.mean(self.fake_s) - torch.mean(self.fake_r), 2) * 0.3) *\
                                      weight.preserve_info_w
        else:
            self.loss_preserve_info = self.Tensor([0.0])

        if weight.identity_w > 0:
            self.loss_idt_s = self.criterion_idt(self.fake_s, self.rec_s.detach()) * weight.lambda_b_w
            self.loss_idt_i = self.IIWReconstLoss(self.fake_r, self.fake_s, self.targets) * weight.lambda_i_w
        else:
            self.loss_idt_s = self.Tensor([0.0])
            self.loss_idt_i = self.Tensor([0.0])

        if weight.ssim_w > 0:
            self.loss_ssim_s = (1 - self.criterion_ssim(self.fake_s, self.rec_s.detach())) * weight.lambda_b_w
        else:
            self.loss_ssim_s = self.Tensor([0.0])

        if weight.gradient_w > 0:
            fake_s_grad, fake_s_gradx, fake_s_grady = get_gradient(self.fake_s.repeat(1,3,1,1))
            self.loss_grad_s = (self.criterion_mse(fake_s_gradx, self.rec_s_gradx) +
                                self.criterion_mse(fake_s_grady, self.rec_s_grady)) * weight.lambda_b_w
        else:
            self.loss_grad_s = self.Tensor([0.0])
        self.loss_grad = self.loss_grad_s

        if weight.divergence_w > 0:
           # 'low':0.0, 'mid':0.05, 'deep':0.25, 'out':0.7
            self.loss_feature_divergence = self.fd_loss * weight.divergence_w
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

        if weight.pixel_suppress_w > 0:
            self.pixel_suppress = self.invalid_pixel_penalty()
        else:
            self.pixel_suppress = self.Tensor([0.0])

        self.loss_gen_idt = self.loss_idt_s + self.loss_idt_i
        # pdb.set_trace()
        self.loss_gen_ssim = self.loss_ssim_s
        self.loss_gen_cross = weight.identity_w * self.loss_gen_idt + \
                              weight.preserve_info_w * self.loss_preserve_info + \
                              weight.ssim_w * self.loss_gen_ssim + \
                              weight.gradient_w * self.loss_grad + \
                              weight.divergence_w * self.loss_feature_divergence + \
                              weight.perspective_w * self.loss_perspective + \
                              weight.pixel_suppress_w * self.pixel_suppress

    def _compute_perspective_loss(self):
        # chromaticity = Variable(self.targets['chromaticity'].cuda(), requires_grad=False)
        # p_R = torch.mul(chromaticity, self.fake_r.repeat(1, 3, 1, 1))
        in_queue_flag = True
        if self.fake_R_pool.num_imgs > 0 and self.fake_S_pool.num_imgs > 0:
            # pdb.set_trace()
            B, C, H, W = self.rec_s.size()
            for b in range(B):
                if torch.mean(self.rec_s[b]) > 1.05 * torch.mean(self.real_i[b]) or \
                   torch.sum(self.rec_s[b] > self.shading_max) / (H * W) > self.threshold or\
                   torch.sum(self.fake_r[b] < self.albedo_min) / (H * W) > self.threshold:
                    in_queue_flag = False

        if in_queue_flag:
            self.fake_r_ref = self.fake_R_pool.query(self.fake_r.detach().repeat(1,3,1,1))
            self.fake_s_ref = self.fake_S_pool.query(self.rec_s.detach().repeat(1,3,1,1))

        fea_fake_r_base = self.gen_decompose.encoder_shared(self.fake_r.repeat(1,3,1,1))
        fea_ref_r_base = self.gen_decompose.encoder_shared(self.fake_r_ref)
        fea_fake_s_base = self.gen_decompose.encoder_shared(self.fake_s.repeat(1,3,1,1))
        fea_ref_s_base = self.gen_decompose.encoder_shared(self.fake_s_ref)

        fea_fake_r = self.gen_decompose.encoder_a(fea_fake_r_base['mid2'])
        fea_fake_s = self.gen_decompose.encoder_b(fea_fake_s_base['mid2'])
        fea_ref_r  = self.gen_decompose.encoder_a(fea_ref_r_base['mid2'])
        fea_ref_s  = self.gen_decompose.encoder_b(fea_ref_s_base['mid2'])

        # fea_fake_r.update(fea_fake_r_base)
        # fea_fake_s.update(fea_fake_s_base)
        # fea_ref_r.update(fea_ref_r_base)
        # fea_ref_s.update(fea_ref_s_base)

        self.loss_perspective_r = self.criterion_perspective(fea_fake_r, fea_ref_r,
                                                             detail_weights=self.opt.optim.div_detail_dict)
        # self.loss_perspective_r = self.Tensor([0.0])
        self.loss_perspective_s = self.criterion_perspective(fea_fake_s, fea_ref_s,
                                                             detail_weights=self.opt.optim.div_detail_dict)

    def update_threshold(self):
        self.threshold = self.threshold * self.thre_decay

    def invalid_pixel_penalty(self):
        weight = self.opt.optim
        [_, _, w, h] = self.fake_r.size()
        # im_pixels out of the threshold, should be suppressed
        # self.albedo_min = 0.03
        # self.shading_max = 0.97

        # for albedo
        penalty_a = torch.sum(torch.pow(F.relu(self.albedo_min - self.fake_r),2)) / (w * h)
        # for shading
        penalty_s = torch.sum(torch.pow(F.relu(self.rec_s - self.shading_max),2)) / (w * h)
        if self.opt.is_debug:
            if penalty_a * weight.lambda_r_w + penalty_s * weight.lambda_b_w > 2.0:
                print('Warning: pixel_penalty larger than 2.0!')
                print('penalty_a:', penalty_a)
                print('penalty_s:', penalty_s)
                print(self.fake_r.size())
                pdb.set_trace()
            else:
                pass
        return penalty_a * weight.lambda_r_w + penalty_s * weight.lambda_b_w
        # pass

    def optimize_parameters(self):
        # forward
        self.gen_decompose.train()
        for _ in range(1):
            self.forward()
            self.gen_update()

    def get_current_errors(self):
        """plain prediction loss"""
        ret_errors = OrderedDict()
        ret_errors['loss_total'] = self.loss_gen_total.cpu().item()
        ret_errors['loss_joint_iiw'] = self.loss_gen_joint.cpu().item()
        ret_errors['idt_S'] = self.loss_idt_s.cpu().item()
        ret_errors['idt_I'] = self.loss_idt_i.cpu().item()
        ret_errors['ssim_S'] = self.loss_ssim_s.cpu().item()
        ret_errors['grad_S'] = self.loss_grad.cpu().item()
        ret_errors['fea_divergence'] = self.loss_feature_divergence.cpu().item()
        ret_errors['perceptive'] = self.loss_perspective.cpu().item()
        ret_errors['preserve_info'] = self.loss_preserve_info.cpu().item()
        ret_errors['pixel_penalty'] = self.pixel_suppress.cpu().item()
        ret_errors['loss_rs'] = self.loss_rs.cpu().item(),
        ret_errors['loss_ss'] = self.loss_ss.cpu().item(),
        ret_errors['loss_iiw'] = self.loss_iiw.cpu().item(),
        return ret_errors

    def evlaute_iiw(self):
        # switch to evaluation mode
        self.gen_decompose.eval()
        total_whdr, total_whdr_eq, total_whdr_ineq, count = self.criterion_joint.evaluate_WHDR(self.fake_r, self.targets)
        self.gen_decompose.train()
        return total_whdr, total_whdr_eq, total_whdr_ineq, count

    def compute_pr(self, pixel_labels_dir, splits_dir, dataset_split, class_weights, bl_filter_size, img_dir, thres_count=400):

        thres_list = saw_utils.gen_pr_thres_list(thres_count)
        photo_ids = saw_utils.load_photo_ids_for_split(
            splits_dir=splits_dir, dataset_split=dataset_split)

        plot_arrs = []
        line_names = []

        fn = 'pr-%s' % {'R': 'train', 'V': 'val', 'E': 'test'}[dataset_split]
        title = '%s Precision-Recall' % (
            {'R': 'Training', 'V': 'Validation', 'E': 'Test'}[dataset_split],
        )

        print("FN ", fn)
        print("title ", title)

        # compute PR
        rdic_list = self.get_precision_recall_list_new(pixel_labels_dir=pixel_labels_dir, thres_list=thres_list,
            photo_ids=photo_ids, class_weights=class_weights, bl_filter_size = bl_filter_size, img_dir=img_dir)

        plot_arr = np.empty((len(rdic_list) + 2, 2))

        # extrapolate starting point
        plot_arr[0, 0] = 0.0
        plot_arr[0, 1] = rdic_list[0]['overall_prec']

        for i, rdic in enumerate(rdic_list):
            plot_arr[i+1, 0] = rdic['overall_recall']
            plot_arr[i+1, 1] = rdic['overall_prec']

        # extrapolate end point
        plot_arr[-1, 0] = 1
        plot_arr[-1, 1] = 0.5

        AP = np.trapz(plot_arr[:,1], plot_arr[:,0])

        return AP

    def get_precision_recall_list_new(self, pixel_labels_dir, thres_list, photo_ids,
                                  class_weights, bl_filter_size, img_dir):

        output_count = len(thres_list)
        overall_conf_mx_list = [
            np.zeros((3, 2), dtype=int)
            for _ in range(output_count)
        ]

        count = 0
        total_num_img = len(photo_ids)

        for photo_id in (photo_ids):
            print("photo_id ", count, photo_id, total_num_img)
            # load photo using photo id, hdf5 format
            img_path = img_dir + str(photo_id) + ".png"

            saw_img = saw_utils.load_img_arr(img_path)
            original_h, original_w = saw_img.shape[0], saw_img.shape[1]
            saw_img = saw_utils.resize_img_arr(saw_img)

            saw_img = np.transpose(saw_img, (2,0,1))
            input_ = torch.from_numpy(saw_img).unsqueeze(0).contiguous().float()
            input_images = Variable(input_.cuda() , requires_grad = False)

            # run model on the image to get predicted shading
            # prediction_S , rgb_s = self.netS.forward(input_images)
            prediction_S, prediction_R, _ = self.gen_decompose(input_images)[:3]
            # prediction_Sr = prediction_S.repeat(1,3,1,1)

            # output_path = root + '/phoenix/S6/zl548/SAW/prediction/' + str(photo_id) + ".png.h5"
            prediction_Sr = torch.exp(prediction_S * 0.4545)

            prediction_S_np = prediction_Sr.data[0,0,:,:].cpu().numpy()
            prediction_S_np = resize(prediction_S_np, (original_h, original_w), order=1, preserve_range=True)

            # compute confusion matrix
            conf_mx_list = self.eval_on_images( shading_image_arr = prediction_S_np,
                pixel_labels_dir=pixel_labels_dir, thres_list=thres_list,
                photo_id=photo_id, bl_filter_size = bl_filter_size, img_dir=img_dir
            )

            for i, conf_mx in enumerate(conf_mx_list):
                # If this image didn't have any labels
                if conf_mx is None:
                    continue
                overall_conf_mx_list[i] += conf_mx

            count += 1

            ret = []
            for i in range(output_count):
                overall_prec, overall_recall = saw_utils.get_pr_from_conf_mx(
                    conf_mx=overall_conf_mx_list[i], class_weights=class_weights,
                )

                ret.append(dict(
                    overall_prec=overall_prec,
                    overall_recall=overall_recall,
                    overall_conf_mx=overall_conf_mx_list[i],
                ))
        return ret

    def eval_on_images(self, shading_image_arr, pixel_labels_dir, thres_list, photo_id, bl_filter_size, img_dir):
        """
        This method generates a list of precision-recall pairs and confusion
        matrices for each threshold provided in ``thres_list`` for a specific
        photo.

        :param shading_image_arr: predicted shading images

        :param pixel_labels_dir: Directory which contains the SAW pixel labels for each photo.

        :param thres_list: List of shading gradient magnitude thresholds we use to
        generate points on the precision-recall curve.

        :param photo_id: ID of the photo we want to evaluate on.

        :param bl_filter_size: The size of the maximum filter used on the shading
        gradient magnitude image. We used 10 in the paper. If 0, we do not filter.
        """

        shading_image_linear_grayscale = shading_image_arr
        shading_image_linear_grayscale[shading_image_linear_grayscale < 1e-4] = 1e-4
        shading_image_linear_grayscale = np.log(shading_image_linear_grayscale)

        shading_gradmag = saw_utils.compute_gradmag(shading_image_linear_grayscale)
        shading_gradmag = np.abs(shading_gradmag)

        if bl_filter_size:
            shading_gradmag_max = maximum_filter(shading_gradmag, size=bl_filter_size)

        # We have the following ground truth labels:
        # (0) normal/depth discontinuity non-smooth shading (NS-ND)
        # (1) shadow boundary non-smooth shading (NS-SB)
        # (2) smooth shading (S)
        # (100) no data, ignored
        y_true = saw_utils.load_pixel_labels(pixel_labels_dir=pixel_labels_dir, photo_id=photo_id)

        img_path = img_dir + str(photo_id) + ".png"

        # diffuclut and harder dataset
        srgb_img = saw_utils.load_img_arr(img_path)
        srgb_img = np.mean(srgb_img, axis=2)
        img_gradmag = saw_utils.compute_gradmag(srgb_img)

        smooth_mask = (y_true == 2)
        average_gradient = np.zeros_like(img_gradmag)
        # find every connected component
        labeled_array, num_features = label(smooth_mask)
        for j in range(1, num_features + 1):
            # for each connected component, compute the average image graident for the region
            avg = np.mean(img_gradmag[labeled_array == j])
            average_gradient[labeled_array == j] = avg

        average_gradient = np.ravel(average_gradient)

        y_true = np.ravel(y_true)
        ignored_mask = y_true > 99

        # If we don't have labels for this photo (so everything is ignored), return
        # None
        if np.all(ignored_mask):
            return [None] * len(thres_list)

        ret = []
        for thres in thres_list:
            y_pred = (shading_gradmag < thres).astype(int)
            y_pred_max = (shading_gradmag_max < thres).astype(int)
            y_pred = np.ravel(y_pred)
            y_pred_max = np.ravel(y_pred_max)
            # Note: y_pred should have the same image resolution as y_true
            assert y_pred.shape == y_true.shape

            # confusion_matrix = saw_utils.grouped_confusion_matrix(y_true[~ignored_mask], y_pred[~ignored_mask], y_pred_max[~ignored_mask])
            confusion_matrix = saw_utils.grouped_weighted_confusion_matrix(y_true[~ignored_mask], y_pred[~ignored_mask],
                                                                           y_pred_max[~ignored_mask],
                                                                           average_gradient[~ignored_mask])
            ret.append(confusion_matrix)

        return ret

    def get_current_visuals(self):
        mean = self.opt.data.image_mean
        std = self.opt.data.image_std
        use_norm = self.opt.data.use_norm

        img_real_i = utils.tensor2img(self.input_i.detach().clone(), mean, std, use_norm=False)

        chromaticity = Variable(self.targets['chromaticity'].cuda(), requires_grad=False)
        p_R = torch.mul(chromaticity, self.fake_r.repeat(1, 3, 1, 1))

        # p_R = self.fake_r.repeat(1, 3, 1, 1)
        # pdb.set_trace()

        img_fake_r = utils.tensor2img(p_R.detach().clone(), mean, std, use_norm=False)

        img_rec_s = utils.tensor2img(self.rec_s.detach().clone(), mean, std, use_norm=False)
        img_fake_s = utils.tensor2img(self.fake_s.detach().clone(), mean, std, use_norm=False)

        ret_visuals = OrderedDict([('real_I', img_real_i),
                                   ('fake_S', img_fake_s),
                                   ('rec_S', img_rec_s),  # reconstructed s from r
                                   ('fake_R', img_fake_r),
                                   ])

        return ret_visuals

    @staticmethod
    def loss_log(losses):
        log_detail = '\t{}:{}\n'.format('loss_total', losses['loss_total']) +\
                     '  {}:{}\n'.format('loss_joint_iiw', losses['loss_joint_iiw']) +\
                     '\t{}:{:.6f}  {}:{:.6f}  {}:{:.6f}  {}:{:.6f}  {}:{:.6f}\n'.format(
                         'idt_I', losses['idt_I'],
                         'idt_S', losses['idt_S'],
                         'ssim_S', losses['ssim_S'],
                         'preserve_R', losses['preserve_info'],
                         'grad_S', losses['grad_S']) + \
                     '\t{}:{}  {}:{}  {}:{}\n'.format('divergence', losses['fea_divergence'],
                                                      'perceptive', losses['perceptive'],
                                                      'pixel_penalty',
                                                      losses['pixel_penalty']) + \
                     '\t{}:{}  {}:{}  {}:{}\n'.format('loss_rs',
                                                      losses['loss_rs'],
                                                      'loss_ss',
                                                      losses['loss_ss'],
                                                      'loss_iiw',
                                                      losses['loss_iiw'])
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
        self.save_network(self.gen_decompose, 'G_decompose', label)

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
        if lr <= 1.5e-6 and self.opt.use_wave_lr:  # reset lr
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

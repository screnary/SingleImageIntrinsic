import torch
import torchvision
import my_data
#from configs.intrinsic_iiw import opt
from configs.intrinsic_iiw_oneway_on_saw import opt
import trainer_saw as Trainer
from utils import save_eval_images_iiw, check_dir, visualize_inspector_iiw
from utils.image_pool import ImagePool
import numpy as np
import pdb
import warnings
warnings.filterwarnings('ignore')

check_dir(opt.output_root)
check_dir(opt.logger.log_dir)


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


def test_SAW(epoch):
    # parameters for SAW
    settings = basic_settings()
    model_trainer = settings['model_trainer']
    model_trainer.resume(model_trainer.gen_decompose, 'G_decompose',
                         epoch_name=epoch)
    full_root = '../'
    pixel_labels_dir = full_root + '/datasets/SAW/saw_pixel_labels/saw_data-filter_size_0-ignore_border_0.05-normal_gradmag_thres_1.5-depth_gradmag_thres_2.0'
    splits_dir = full_root + '/datasets/SAW/saw_splits/'
    img_dir = full_root + "/datasets/SAW/saw_images_512/"
    dataset_split = 'E'
    class_weights = [1, 1, 2]
    bl_filter_size = 10

    print("============================= Validation ON SAW============================")
    model_trainer.gen_decompose.eval()
    AP = model_trainer.compute_pr(pixel_labels_dir, splits_dir,
                          dataset_split, class_weights, bl_filter_size, img_dir)

    print("Current AP: %f" % AP)
    model_trainer.gen_decompose.train()
    return AP


if __name__ == '__main__':
    assert torch.cuda.is_available()
    AP = test_SAW(12)

    # best_whdr, best_epoch = train()
    # print('whdr=', best_whdr)
    # print('best ep:', best_epoch)
    # test(epoch=best_epoch)
    # test(epoch=20)

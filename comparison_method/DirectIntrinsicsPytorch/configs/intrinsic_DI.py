import easydict as ed
import os
##########################
# Root options-1
##########################
opt = ed.EasyDict()
opt.use_wave_lr = True
opt.is_train = True                         # The flag to determine whether it is train or not [True/False]
opt.continue_train = True                   # The flag to determine whether to continue training
opt.which_epoch = 100                  # Which epoch to recover
opt.gpu_ids = [0]                           # gpu ids, [0/2/3] or their combination
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
##########################
# Data options
##########################
opt.data = ed.EasyDict()

opt.data.name = 'MPI-main-RD'                               # [MPI-main] the name of dataset [MPI, MIT] MPI-main
opt.data.split = 'sceneSplit'                       # [imageSplit, sceneSplit]
opt.data.cropped = False                             # if is True, there is no need further crop in data transform
opt.data.cropped_test = False
opt.data.input_dim_a = 3                            # number of image channels [1/3]
opt.data.input_dim_b = 3                            # number of image channels [1/3]
opt.data.input_dim_c = 3                            # number of image channels [1/3]
opt.data.num_workers = 8                            # number of data loading threads
opt.data.use_norm = True  #True                            # whether use normalization [True/False]
opt.data.batch_size = 4                             # batch size [8]
opt.data.batch_size_test = 1
opt.data.load_size = 436                            # loading image size
opt.data.new_size = 384                             # the output of dataset's image size, MSCR method need size tobe multiple of 32
opt.data.new_size_test = 256                        # [304] for test image size
opt.data.no_flip = False                            # the flag to determine not use random flip[True/False]
opt.data.unpaired = False
opt.data.serial_batches = True                      # take images in order or not [True/False], not shuffle while create
opt.data.is_train = opt.is_train                    # the flag to determine whether it is train or not
opt.data.data_root = '../datasets/'        # dataset folder location
opt.data.preprocess = 'resize_and_crop_refine_RD'             # pre-process[resize_and_crop/crop/scale_width/scale_width_and_crop] [resize_and_crop]
opt.data.image_mean = (0., 0., 0.)      # image mean value  (0.4914, 0.4822, 0.4465), (0.5, 0.5, 0.5)
opt.data.image_std = (1., 1., 1.)       # image standard difference value  (0.2023, 0.1994, 0.2010)


##########################
# Trainer options
##########################
opt.train = ed.EasyDict()

opt.train.mode = 'direct_intrinsics'                 # train generator mode [self-sup/none], [self-sup/cross-deconv/self-sup-ms]
opt.train.trainer_mode = 'Basic'            # [Basic/GAN]
opt.train.pool_size = 100                   # image pool size
opt.train.print_freq = 100                  # frequency of showing training results on console
opt.train.display_freq = 100                # frequency of showing training results in the visdom
opt.train.update_html_freq = 500            # frequency of showing training results in the web html
opt.train.save_latest_freq = 1000           # frequency of saving the latest results
opt.train.save_epoch_freq = 4               # the starting epoch count: epoch_count += save_freq
opt.train.n_iter = 100                      # iterations at starting learning rate
opt.train.n_iter_decay = 100                # iterations to linearly decay learning rate to zero

opt.train.save_train_img = True
opt.train.save_per_n_ep = 5  # 10
opt.train.save_per_n_ep_train = 5  # 10
opt.train.epoch_count = 1 if not opt.continue_train else 1+opt.which_epoch                  # the starting epoch count: epoch_count += save_freq
opt.train.total_ep = 150  # 250, MIT-50

##########################
# Model options
##########################
opt.model = ed.EasyDict()

# generator options
opt.model.gen = ed.EasyDict()

opt.model.gen.input_dim = 3                 # number of dimension in input image
opt.model.gen.output_dim_r = 3              # number of dimension in background layer
opt.model.gen.output_dim_s = 1  # 1              # number of dimension in residual layer
opt.model.gen.pad_type = 'reflect'          # padding type [zero/reflect]
opt.model.gen.norm = 'in'                   # normalization layer [none/bn/in/ln], before[in]

# discriminator options

##########################
# Root options-2
##########################
# The root dir for saving trained parameters and log information
if 'MPI' in opt.data.name:
    opt.output_root = '../ckpoints-'+opt.train.trainer_mode+'-'+opt.train.mode+'-'+opt.data.name+'-'+opt.data.split
else:
    opt.output_root = '../ckpoints-' + opt.train.trainer_mode + '-' + opt.train.mode + '-' + opt.data.name

##########################
# Optimization options
##########################
opt.optim = ed.EasyDict()

opt.optim.max_iter = 1000000                # maximum number of training iterations
opt.optim.weight_decay = 1e-6             # weight decay
opt.optim.n_iter_decay = 50  # 30                # iterations to linearly decay learning rate to zero [100]
opt.optim.beta1 = 0.0                       # Adam parameter
opt.optim.beta2 = 0.9                       # Adam parameter
opt.optim.init = 'kaiming'                  # initialization [gaussian/kaiming/xavier/orthogonal]
opt.optim.lr_g = 0.0001                     # initial learning rate for generator, [0.0001]
opt.optim.lr_policy = 'step'                # learning rate scheduler
opt.optim.step_size = 6000                # how often to decay learning rate
opt.optim.gamma = 0.5                       # how much to decay learning rate
opt.optim.epoch_count = 1                   # the starting epoch count: epoch_count += save_freq
opt.optim.cycle_w = 100                     # weight of cycle consistency loss

opt.optim.identity_w = 10.0                  # weight of image identity loss
opt.optim.gradient_w = 1.5  #1.5                  # weight of image gradient loss, [1.0]

opt.optim.lambda_i_w = 0.0                  # weight for cycle loss (I -> B, R -> I), [default 10.0] [1.0]
opt.optim.lambda_b_w = 5.0                  # weight for cycle loss (B, R -> I -> B, R) B
opt.optim.lambda_r_w = 5.0                  # weight for cycle loss (B, R -> I -> B, R) R [5.0]

##########################
# Logger options
##########################
opt.logger = ed.EasyDict()

opt.logger.image_save_iter = 10000          # How often do you want to save output images during training
opt.logger.image_display_iter = 100         # How often do you want to display output images during training
opt.logger.display_size = 16                # How many images do you want to display each time
opt.logger.snapshot_save_iter = 10000       # How often do you want to save trained models
opt.logger.log_iter = 10                    # How often do you want to log the training stats
opt.logger.log_dir = opt.output_root+'/log/'      # The log dir for saving train log and image information
opt.logger.root_dir = opt.output_root       # The root dir for logging
opt.logger.is_train = opt.is_train          # Copy the `is_train` flag
opt.logger.display_id = 1                   # Window id of the web display
opt.logger.display_single_pane_ncols = 0    # If positive, display all images in a single visdom with certain cols'
opt.logger.no_html = False                  # do not save intermediate training results to web
opt.logger.display_port = 8097              # visdom port of the web display

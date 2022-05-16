%%% RUN Relative Layer on MIT dataset %%%
%% Step-1: query dataset

DATA_ROOT = 'D:/repo/dataset/MIT';
FILE_MODE = 'train';
SUB_DIR = 'MIT-input';

image_name_lst = {};
lines = readlines([DATA_ROOT '/' FILE_MODE '.txt']);
n_images = length(lines);

for k1 = 1:n_images
    image_name_lst{k1} = [DATA_ROOT '/' SUB_DIR '/' lines{k1}];
end

%% Step-3: extract R, L of input image and save them
SAVE_DIR = [DATA_ROOT '/Result-' SUB_DIR '-' FILE_MODE '-RelativeSmooth'];
if ~exist(SAVE_DIR, 'dir')
    mkdir(SAVE_DIR);
end

for k1 = 1:n_images
    f_name = image_name_lst{k1};
    fprintf('## Processing Image [%d / %d] %s ... \n', k1, n_images, f_name);
    I = imread(f_name);
    I1 = im2double(I);
    [R, S] = Intrinsic_Relsmo(I1, 2);
    
    t_name_elems = split(lines{k1}, '.');
    imwrite(R, [SAVE_DIR '/' t_name_elems{1} '-R.png'])
    imwrite(S, [SAVE_DIR '/' t_name_elems{1} '-S.png'])
end
clear all;
close all;

DataSet = 'MPI';

DataDir = ['../datasets/',DataSet,'/'];
DumpDir = ['../datasets/', DataSet, '/temp_refine_2/'];
RD_Dir = ['../datasets/', DataSet, '/refined_gs/'];
DataDir = RD_Dir;

InputDir = [DataDir, 'MPI-main-clean/'];
AlbedoDir = [DataDir, 'MPI-main-albedo/'];
ShadingDir = [DataDir, 'MPI-main-shading/'];

sintel_scenes = { 'alley_1', 'bamboo_1', 'bandage_1', 'cave_2', 'market_2', 'market_6', 'shaman_2', 'sleeping_1', 'temple_2',...
    'alley_2', 'bamboo_2', 'bandage_2', 'cave_4', 'market_5', 'mountain_1', 'shaman_3', 'sleeping_2', 'temple_3'};

mse_tab = [];
mse_tab_all = [];
for sid = 1:length(sintel_scenes)
    scene = sintel_scenes{sid};
    disp(scene);
    file_lists = dir([ShadingDir,scene '*.png']);
    mse_tab_scene = [];
    for n = 1:length(file_lists)
        frame_name = file_lists(n).name;
        albedoName = [AlbedoDir frame_name];
        shadingName = [ShadingDir frame_name];
        inputName = [InputDir frame_name];
        
        input = im2double(imread(inputName));
        albedo = im2double(imread(albedoName));
        shading = im2double(imread(shadingName));
        input_rec = albedo .* shading;
        
        [height, width, channel] = size(albedo);
        
        mse_img = sqrt(sum((input - input_rec).^2, 3));
        mse = sum(mse_img(:)) / (height*width);
        mse_tab_scene = [mse_tab_scene; mse];
        mse_tab_all = [mse_tab_all; mse];
    end
    mse_tab = [mse_tab; mean(mse_tab_scene(:))];
end

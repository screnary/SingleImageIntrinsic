%%%% function masks = propagate_invalid_masks()
%%%% using optical flow to propagate invalid masks in the same scene
%%%% need to run before MPI_data_prepare.m, to get the LLE inpainting masks
%%%% specify the invalid region for each synset
clear all;
close all;

run synset_settings;

% synnames = {'market_2', 'market_5', 'sleeping_2'};
% regions = {[600,830;1,80], [1,1; 1,1], [450,550;225,325]};

synnames = {'alley_1',
'alley_2',
'bamboo_1',
'bamboo_2',
'bandage_1',
'bandage_2',
'cave_2',
'cave_4',
'market_2',
%'market_5',
'market_6',
'mountain_1',
'shaman_2',
'shaman_3',
'sleeping_1',
'sleeping_2',
'temple_2',
'temple_3'};

for syn_i = 1:length(synnames)
    synname = synnames{syn_i};
    start_id = 1;
    count = 50;
    for syn_id = 1:size(synsets,1)
        if strcmp(synname, synsets{syn_id, 1})
            start_id = synsets{syn_id, 2};
            count = synsets{syn_id, 3};
            break;
        end
    end

    DataSet = 'MPI';

    DataDir = ['../../data/',DataSet,'/origin/'];
    DumpDir = ['../../data/', DataSet, '/temp_refine_2/'];

    MaskDir = [DataDir, 'MPI-main-mask/'];
    InputDir = [DataDir, 'MPI-main-clean/'];
    AlbedoDir = [DataDir, 'MPI-main-albedo/'];
    ShadingDir = [DataDir, 'MPI-main-shading/'];
    flowDir = ['../../data/MPI-Sintel-complete/training/flow/', synname, '/'];
    outflowDir = ['../../data/', DataSet, '/flow_hdf5/'];

    file_lists = dir([ShadingDir, '*.png']);
    disp([synname, '   ', length(file_lists)])

    valid_masks = {};
    invalid_masks = {};
    flow_images = {};
    for iter = 0:(count-1) %451:500 %591:640, 741:790 %length(file_lists)
       %  % get the optical images
        if iter < count-1
            f_path = [flowDir, 'frame_', num2str(iter+1, '%04d'), '.flo'];
            flo_data = readFlowFile(f_path);

            flow_images{iter+1} = flo_data;
            h5_file = [outflowDir, synname, '_frame_', num2str(iter+1, '%04d'), '.h5'];
            h5create(h5_file, '/data', size(flo_data));
            h5write(h5_file, '/data', flo_data);
        end
    end
end

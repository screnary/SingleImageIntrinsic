% % % check the refined_gray scale shading, for I = A.*S formula
% % % step 4: statistics of refined shadings
% % % step 5: aligned after python code align_image_pixel.py, regenerate
% % %         clean-pass images

clear all;
close all;

DataSet = 'MPI';

DataDir = ['../datasets/',DataSet,'/origin/'];
OutDir = ['../datasets/', DataSet, '/refined_final/'];
OutDir_new = ['../datasets/', DataSet, '/refined_final/'];

% OutDir = DataDir;
InShadingDir = [DataDir, 'MPI-main-shading/'];

MaskDir = [DataDir, 'MPI-main-mask/'];
InputDir = [OutDir, 'MPI-main-clean/'];
AlbedoDir = [OutDir, 'MPI-main-albedo/'];
ShadingDir = [OutDir, 'MPI-main-shading/'];
file_lists = dir([MaskDir, '*.png']);  %(890 imgs)

run synset_settings;

for syn_i = 1:size(synsets,1)
    synname = synsets{syn_i, 1};
    start_id = synsets{syn_i, 2};
    count = synsets{syn_i, 3};


    disp(['processing synname: ', synname]);
    for iter = 0:(count-1)
        n = iter + start_id;
        frame_name = file_lists(n).name;

        albedoName = [AlbedoDir frame_name];
        shadingName = [ShadingDir frame_name];
        maskName = [MaskDir frame_name];
        inputName = [InputDir frame_name];

        albedo = im2double(imread(albedoName));
        shading = im2double(imread(shadingName));
        [height, width, channel] = size(albedo);
        maskimg = repmat(imresize(imread(maskName), [height, width], 'nearest'),[1,1,3]);
        valid_idx = maskimg == 255;

        %% simple assumption: I = A.*S
        albedo(albedo == 0) = 0.01;  % eleminate absolute zeros

        %% L*a*b* color space
%         input_lab = rgb2lab(input);
        albedo_lab = rgb2lab(albedo);
        shading_lab = rgb2lab(shading);

%         L_i = input_lab(:,:,1);  % the Luminance channel, [0,100]
        L_a = albedo_lab(:,:,1);
%         L_s = shading_lab(:,:,1);
        L_s = shading;

        %% rec type 1: there will be flaws in Albedo
    %     input_new = input;
    %     shading_new = repmat(L_s / 100, [1,1,3]);
    %     albedo_new = input_new ./ shading_new;

        %% rec type 2
        albedo_new = albedo;
        L_s(L_s == 0) = 0.03;
%         shading_new = repmat(L_s / 100, [1,1,3]);
        shading_new = shading;
        input_new = albedo_new .* shading_new;
        input_new(input_new == 0) = 0.013;

        if sum(L_s(:) == 0) > 0
            disp('Warning! shading has zero values!')
            figure(1), imshow([input_new; albedo_new; shading_new])
            figure(2), imshow([albedo, input_new ./ shading_new; shading, input_new ./ albedo_new])
        end

        albedo_invalid_mask = albedo(:,:,1)==0 | albedo(:,:,2)==0 | albedo(:,:,3)==0;
        if sum(albedo_invalid_mask(:))
            disp('Warning! albedo has zero values!')
            figure(3), imshow(albedo_invalid_mask)
        end

        input_invalid_mask = input_new(:,:,1)==0 | input_new(:,:,2)==0 | input_new(:,:,3)==0;
        if sum(input_invalid_mask(:))
            disp('Warning! input has zero values!')
            figure(3), imshow(input_invalid_mask)
        end


        %% save files
        imwrite(input_new, [OutDir_new, 'MPI-main-clean/', frame_name]);
%         imwrite(albedo_new, [OutDir_new, 'MPI-main-albedo/', frame_name]);
%         imwrite(shading_new, [OutDir_new, 'MPI-main-shading/', frame_name]);
        
    end

    %% process the jittered frames

end
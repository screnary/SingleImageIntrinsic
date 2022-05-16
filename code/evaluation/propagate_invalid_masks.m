%%%% function masks = propagate_invalid_masks()
%%%% using optical flow to propagate invalid masks in the same scene
%%%% need to run before MPI_data_prepare.m, to get the LLE inpainting masks
%%%% specify the invalid region for each synset
clear all;
close all;

run synset_settings;

% synnames = {'market_2', 'market_5', 'sleeping_2'};
% regions = {[600,830;1,80], [1,1; 1,1], [450,550;225,325]};
synnames = {'market_5'};
regions = {[220,1021;1,420]};
for syn_i = 1:length(synnames)
    synname = synnames{syn_i};
    region = regions{syn_i};
    x_range = region(1,:);
    y_range = region(2,:);
    start_id = 1;
    count = 50;
    for syn_id = 1:size(synsets,1)
        if strcmp(synname, synsets{syn_id, 1})
            start_id = synsets{syn_id, 2};
            count = synsets{syn_id, 3};
            break;
        end
    end

    DataSet = 'MPI/origin';

    DataDir = ['../datasets/',DataSet,'/'];
    DumpDir = ['../datasets/', DataSet, '/temp_refine_2/'];
    OutDir = ['../datasets/', DataSet, '/refined_test/'];

    MaskDir = [DataDir, 'MPI-main-mask/'];
    InputDir = [DataDir, 'MPI-main-clean/'];
    AlbedoDir = [DataDir, 'MPI-main-albedo/'];
    ShadingDir = [DataDir, 'MPI-main-shading/'];
    flowDir = ['../datasets/MPI-Sintel-complete/training/flow/', synname, '/'];

    file_lists = dir([ShadingDir, '*.png']);

    valid_masks = {};
    invalid_masks = {};
    flow_images = {};
    for iter = 0:(count-1) %451:500 %591:640, 741:790 %length(file_lists)
        n = iter + start_id;
        gap = 6.0;
        threshold = 0.985;
        scale_factor1 = 1.5;
        scale_factor2 = 1.5;
        lle_flag = true;
        if n > 200 && n <= 300  % bandage
            threshold = 0.98;
            scale_factor2 = 6.7;
        elseif n > 300 && n <= 400  % cave, girl slash dragon
            threshold = 0.98;
            gap = 7.0;
            scale_factor1 = 3.5;
        elseif n > 540 && n <= 590  % mountain
            lle_flag = false;
        elseif n > 590 && n <= 640
            threshold = 0.95;
        elseif n > 640 && n <= 690  % grase and stample
            threshold = 0.98;
            scale_factor2 = 6.5;
        elseif n > 690 && n <= 740  % sleeping 1
            threshold = 0.955;
            scale_factor1 = 5.5;
        elseif n > 741 && n <= 790  % sleeping 2
            threshold = 0.97;
            scale_factor2 = 2.5;
        elseif n > 790 && n <= 840  % temple 2
            scale_factor1 = 1.5;
            threshold = 0.985;
        else
        end

        frame_name = file_lists(n).name;
        albedoName = [AlbedoDir frame_name];
        shadingName = [ShadingDir frame_name];
        maskName = [MaskDir frame_name];
        inputName = [InputDir frame_name];

        disp(['...Processing: ', num2str(n), '; FileName: ', frame_name]);

        input = im2double(imread(inputName));
        albedo = im2double(imread(albedoName));
        shading = im2double(imread(shadingName));
        [height, width, channel] = size(albedo);
        maskimg = repmat(imresize(imread(maskName), [height, width], 'nearest'),[1,1,3]);
        valid_idx = maskimg == 255;

        %% simple assumption: I = A.*S
        input_rec = albedo .* shading;
        albedo_rec = input ./ (shading);
        shading_rec = input ./ (albedo);

        %% L*a*b* color space
        input_lab = rgb2lab(input);
        albedo_lab = rgb2lab(albedo);
        shading_lab = rgb2lab(shading);
        input_rec_lab = rgb2lab(input_rec);
        albedo_rec_lab = rgb2lab(albedo_rec);
        shading_rec_lab = rgb2lab(shading_rec);

        L_i = input_lab(:,:,1);  % the Luminance channel, [0,100]
        L_a = albedo_lab(:,:,1);
        L_s = shading_lab(:,:,1);
        L_i_rec = input_rec_lab(:,:,1);
        L_a_rec = albedo_rec_lab(:,:,1);
        L_s_rec = shading_rec_lab(:,:,1);

        L_i(L_i < 4) = 4;
        input_lab_filtered = input_lab;
        input_lab_filtered(:,:,1) = L_i;
        input_filtered = lab2rgb(input_lab_filtered);  % remove zero pixels !

        L_s(L_s < 4) = 4;
        shading_lab_filtered = shading_lab;
        shading_lab_filtered(:,:,1) = L_s;
        shading_filtered = lab2rgb(shading_lab_filtered);  % remove zero pixels !
    %     figure(1), imshow(shading)
        % get validate pixels in L_albedo_rec
        a_valid_mask = L_a_rec < 100 & L_a_rec > 4;  % use this as valid
        s_valid_mask = L_s_rec < 95;   % [95] or use this. more strict

        % get the statistics of albedo
        L_arec_valid = L_a_rec(s_valid_mask & a_valid_mask);
        mu_arec = mean(L_arec_valid);
        std_arec = std(L_arec_valid);

        mu_a = mean(L_a(:));
        std_a = std(L_a(:));

        % distribution shift
        L_a_shift = (L_a - mu_a) * 0.99*std_arec / std_a + 1.01*mu_arec;
        if min(L_a_shift(:)) <= 0
            L_a_shift_normed = 100 * (L_a_shift - min(L_a_shift(:))+1) / (max(L_a_shift(:))+1 - min(L_a_shift(:))/scale_factor1);
        else
            L_a_shift_normed = 100 * (L_a_shift - min(L_a_shift(:))/scale_factor2+1) / (max(L_a_shift(:))+1 - 0);
        end

        if n > 300 && n <= 400  % cave, girl slash dragon
            L_a_shift_normed(L_a<4) = 35;
        end

        L_s_ = 100 * L_i ./ L_a_shift_normed;
    %     hs = histogram(L_s_);
        %% check invalid values in L_shading_rec

    %     if sum(sum(L_s_ > 100))  % if there are invalid values in L_s_
        [N, edges] = histcounts(L_s_);
        N_cum = cumsum(N);
        percent = N_cum ./ (height*width);
        for j = 1:length(N)
            if percent(j) > threshold % 0.999, 0.98
                up_bound = 0.5 * (edges(j)+edges(j+1));
                break;
            end
        end
        invalid_mask = L_s_ >= up_bound | L_s_ <= 0;
        invalid_masks{iter+1} = invalid_mask;
        valid_masks{iter+1} = s_valid_mask & a_valid_mask;

        %% get the optical images
        if iter < count-1
            f_path = [flowDir, 'frame_', num2str(iter+1, '%04d'), '.flo'];
            flo_data = readFlowFile(f_path);
            flow_images{iter+1} = flo_data;
        end
    end

    %% process and merge the invalid MASKS
    new_masks = {};
    new_masks{1} = invalid_masks{1};
    new_valids = {};
    new_valids{1} = valid_masks{1};
    for iter = 1:length(invalid_masks)-1
        cur_mask = new_masks{iter};
        next_mask = invalid_masks{iter+1};
        o_flow = flow_images{iter};
        x_delta = o_flow(:,:,1);
        y_delta = o_flow(:,:,2);
        cur_idx = find(cur_mask);
        next_idx = find(next_mask);
        [cur_y, cur_x] = ind2sub(size(cur_mask), cur_idx);
        [next_y, next_x] = ind2sub(size(cur_mask), next_idx);
        new_y = int32(cur_y + y_delta(cur_idx));
        new_x = int32(cur_x + x_delta(cur_idx));
        new_sub = [new_y, new_x];
%         new_sub = new_sub(new_y > 0 & new_y <= size(cur_mask,1) & new_x > 0 & new_x <= size(cur_mask, 2), :);
        new_sub = new_sub(new_y > y_range(1) & new_y <= y_range(2) & new_x > x_range(1) & new_x <= x_range(2), :);
        new_idx = sub2ind(size(cur_mask), new_sub(:,1), new_sub(:,2));
        res_mask = next_mask;
        res_mask(new_idx) = 1;
        new_masks{iter+1} = res_mask;

        % for valid masks
        cur_mask_v = new_valids{iter};
        next_mask_v = valid_masks{iter+1};
        cur_idx_v = find(~cur_mask_v);
        next_idx_v = find(~next_mask_v);
        [cur_y_v, cur_x_v] = ind2sub(size(cur_mask), cur_idx_v);
        [next_y_v, next_x_v] = ind2sub(size(cur_mask), next_idx_v);
        new_y = int32(cur_y_v + y_delta(cur_idx_v));
        new_x = int32(cur_x_v + x_delta(cur_idx_v));
        new_sub = [new_y, new_x];
        new_sub = new_sub(new_y > 0 & new_y <= size(cur_mask,1) & new_x > 0 & new_x <= size(cur_mask, 2), :);
        new_idx_v = sub2ind(size(cur_mask), new_sub(:,1), new_sub(:,2));
        res_mask_v = next_mask_v;
        res_mask_v(new_idx_v) = 0;
        res_mask_v(new_idx) = 0;
%         new_valids{iter+1} = res_mask_v;
        new_valids{iter+1} = next_mask_v;
    end

    for iter = 0:length(invalid_masks)-2
        cur_mask = new_masks{length(invalid_masks)-iter};
        last_mask = invalid_masks{length(invalid_masks)-iter-1};
        o_flow = flow_images{length(invalid_masks)-iter-1};
        x_delta = o_flow(:,:,1);
        y_delta = o_flow(:,:,2);
        cur_idx = find(cur_mask);
        last_idx = find(last_mask);
        [cur_y, cur_x] = ind2sub(size(cur_mask), cur_idx);
        [last_y, last_x] = ind2sub(size(last_mask), last_idx);
        new_y = int32(cur_y - y_delta(cur_idx));
        new_x = int32(cur_x - x_delta(cur_idx));
        new_sub = [new_y, new_x];
%         new_sub = new_sub(new_y > 0 & new_y <= size(cur_mask,1) & new_x > 0 & new_x <= size(cur_mask, 2), :);
        new_sub = new_sub(new_y > y_range(1) & new_y <= y_range(2) & new_x > x_range(1) & new_x <= x_range(2), :);
        new_idx = sub2ind(size(cur_mask), new_sub(:,1), new_sub(:,2));
        res_mask = last_mask;
        res_mask(new_idx) = 1;
        new_masks{length(invalid_masks)-iter-1} = res_mask;

    %     % for valid masks
    %     cur_mask_v = new_valids{length(invalid_masks)-iter};
    %     last_mask_v = valid_masks{length(invalid_masks)-iter-1};
    %     cur_idx_v = find(~cur_mask_v);
    %     next_idx_v = find(~next_mask_v);
    %     [cur_y_v, cur_x_v] = ind2sub(size(cur_mask), cur_idx_v);
    %     [next_y_v, next_x_v] = ind2sub(size(cur_mask), next_idx_v);
    %     new_y = int32(cur_y_v + y_delta(cur_idx_v));
    %     new_x = int32(cur_x_v + x_delta(cur_idx_v));
    %     new_sub = [new_y, new_x];
    %     new_sub = new_sub(new_y > 0 & new_y <= size(cur_mask,1) & new_x > 0 & new_x <= size(cur_mask, 2), :);
    %     new_idx_v = sub2ind(size(cur_mask), new_sub(:,1), new_sub(:,2));
    %     res_mask_v = last_mask_v;
    %     res_mask_v(new_idx_v) = 0;
    %     res_mask_v(new_idx) = 0;
    %     new_valids{length(invalid_masks)-iter-1} = res_mask_v;
    end
    
    % dilation for specific region
    % for market_5, frame 1 ~ 22 dilation, otheres remain unchanged
    for iter = 1:18%1:length(invalid_masks)
        cur_mask = new_masks{iter};
        SE = strel('square', 3);  %11, image dilation, 5
        mask_dilated = imdilate(cur_mask, SE);
        cur_mask(y_range(1):y_range(2), x_range(1):x_range(2)) = mask_dilated(y_range(1):y_range(2), x_range(1):x_range(2));
        new_masks{iter} = cur_mask;
    end

    %% save files
    save_shading_dir = ['./shading_masks/',synname,'_dilated.mat'];
    save(save_shading_dir, 'new_masks');

    save_albedo_dir = ['./albedo_masks/',synname,'_dilated.mat'];
    save(save_albedo_dir, 'new_valids');
end
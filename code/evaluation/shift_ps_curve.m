function [im, mask] = shift_ps_curve(img, i_before, i_after)
%%%% change the histogram, instance value i_before are shifted to i_after, 
%%%% and the other instances are shifted linearly
    [m, n] = size(img);
    mask = zeros(m,n);
    img_uint = uint8(img);
    scale = max(img_uint(:));
    ps = zeros(1,scale+1);
    delta = double(i_before) - double(i_after);
    shift = [-(delta):-1,0,1:(delta)];
    kernel = [-(delta):-1, 0, -1:-1:-(delta)];
%     kernel_w = exp(0.01*kernel);
%     deltas = delta * kernel_w;
    deltas = (-1/delta) * kernel.^2 + delta;
    for k=1:length(shift)
        indices = find(img_uint==(i_before+shift(k)));
        selected = ~mask(indices);
        valid_idx = indices(selected);
        img(valid_idx) = img(valid_idx) - deltas(k);
        mask(indices) = mask(indices) + 1;  % record if modified
    end
    im = img;
end
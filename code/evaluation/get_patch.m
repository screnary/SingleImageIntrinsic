function patchIndex = get_patch(sz, w, h, pixind)
%%%% change the patch of size [w,h] (odd number) centered from location pixind in image of
%%%% size sz=[436,1024]
    assert(h<sz(1) && w<sz(2));
    mask = zeros(sz);
    [pix_r, pix_c] = ind2sub(sz, pixind);  % pixel row, col
    % get the patch region [top:bot, left:right]
    top = pix_r - (h-1)/2;
    bot = pix_r + (h-1)/2;
    if top<=0
        top = 1;
        bot = top + h - 1;
    elseif bot > sz(1)
        bot = sz(1);
        top = bot - h + 1;
    end
    left = pix_c - (w-1)/2;
    right = pix_c + (w-1)/2;
    if left <= 0
        left = 1;
        right = left + w - 1;
    elseif right > sz(2)
        right = sz(2);
        left = right - w + 1;
    end
    mask(top:bot, left:right) = 1;
    patchIndex = find(mask);
end
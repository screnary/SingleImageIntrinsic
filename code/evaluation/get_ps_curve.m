function ps = get_ps_curve(img, scale)

[m, n] = size(img);
ps = zeros(1,scale+1);
for k=0:scale
    ps(k+1) = length(find(img==k))/(m*n);
end
end

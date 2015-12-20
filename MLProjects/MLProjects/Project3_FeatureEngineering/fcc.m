function [cc]=fcc(filename)

f = imread(filename);

h=fspecial('average',9);
g=imfilter(f,h,'replicate');
g=im2bw(g,0.5)
figure
imshow(g)
B = boundaries(g);
 
d = cellfun('length', B);
[max_d, k ] = max(d);
b = B{1};

[M N] = size(g);
g = bound2im(b, M, N, min(b(:, 1) ) , min(b(:, 2)));
figure
imshow(g)
[s, su]= bsubsamp(b,2);

g2 = bound2im(s, M, N, min(b(:, 1) ) , min(b(:, 2)));
figure
imshow(g2)

cn=connectpoly(s(:,1),s(:,2));
g2 = bound2im(cn, M, N, min(b(:, 1) ) , min(b(:, 2)));
figure
imshow(g2)

c=fchcode(su);
cc=c.fcc;

end

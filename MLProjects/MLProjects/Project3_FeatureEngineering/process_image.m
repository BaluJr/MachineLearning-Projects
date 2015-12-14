function features = process_image(base_path, img_id, b)

% Read the raw image and mask of interesting area.
pure_png = imread(strcat(base_path, sprintf('%04d', img_id), '_raw.tif'));
mask_png = imread(strcat(base_path, sprintf('%04d', img_id), '_msk.png'));

% Compute histogram features.
sig_1D = sig_1D_signature(mask_png);
phog_feat = (anna_phog(rgb2gray(pure_png), 8, 360, 3, [1;78;1;78]))';

% Normalize the histograms.
phog_feat = phog_feat / sum(phog_feat);
sig_1D = sig_1D / sum(sig_1D);

%intensity histograms
grey = gscale(pure_png, 'full8');
[hist,edges] = histcounts(grey,b);


features = [phog_feat, sig_1D, hist];

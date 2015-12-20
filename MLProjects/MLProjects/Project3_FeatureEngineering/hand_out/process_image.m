function features = process_image(base_path, img_id, b, segbins, fcc)

% Read the raw image and mask of interesting area.
pure_png = imread(strcat(base_path, sprintf('%04d', img_id), '_raw.tif'));
mask_png = imread(strcat(base_path, sprintf('%04d', img_id), '_msk.png'));
masked_png = imread(strcat(base_path, sprintf('%04d', img_id), '_seg.png'));

% Compute given histogram features.
sig_1D = sig_1D_signature(mask_png);
phog_feat = (anna_phog(rgb2gray(pure_png), 8, 360, 3, [1;78;1;78]))';
% Normalize the histograms.
phog_feat = phog_feat / sum(phog_feat);
sig_1D = sig_1D / sum(sig_1D);


%intensity histograms
if (b > 0)
    grey = gscale(masked_png, 'full8');
    h = histcounts(grey, b);
    exact = histcounts(grey,256);
    h(1) = h(1) - exact(1);
    h(b) = h(b) - exact(256);
else
    grey = gscale(pure_png, 'full8');
    h = histcounts(grey, b);
end


% Put all together
if (b > 0)
    if (fcc)
        features = [phog_feat, sig_1D, h, f];
    else
        features = [phog_feat, sig_1D, h];
    end
else
    if (fcc)
        features = [phog_feat, sig_1D, f];
    else
        features = [phog_feat, sig_1D];
    end
end

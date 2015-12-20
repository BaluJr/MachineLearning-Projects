clear all
close all
addpath('./PHOG')
addpath('./DIPUMToolboxV1.1.3')

SEGMENTED_INTENSITY = false;
FCC = false;
for INTENSITY_BINS=10:30:100
    
%%
% Adjust this number as you change the number of features.
NUM_FEATURES = 696+INTENSITY_BINS;

%% Generate 'train.csv'.

% Read the labels for the samples.
train_labels = csvread('train_labels.csv');

training_data = zeros(length(train_labels), NUM_FEATURES + 2);
for i = 1:length(train_labels)
    did = train_labels(i, 1);
    label = train_labels(i, 2);
    
    features = process_image('images/', did, INTENSITY_BINS, SEGMENTED_INTENSITY, FCC);
    
    training_data(i, :) = [did, features, label];
end
%csvwrite(sprintf('train_bins%d.csv',b), training_data);

%% Generate 'test_validate.csv'.

train_ids = train_labels(:, 1);

test_validate_data = zeros(382, NUM_FEATURES + 1);

i = 1;
for did = 1:1272
    if any(train_ids == did)
        continue;
    end
  
    features = process_image('images/', did, INTENSITY_BINS, SEGMENTED_INTENSITY, FCC);

    test_validate_data(i, :) = [did, features];
    i = i + 1;
end


if (SEGMENTED_INTENSITY)
    if (FCC)
        csvwrite(sprintf('test_validate_segbins%d_fcc.csv',INTENSITY_BINS), test_validate_data);
        csvwrite(sprintf('train_segbins%d_fcc.csv',INTENSITY_BINS), training_data);
    else
        csvwrite(sprintf('test_validate_segbins%d.csv',INTENSITY_BINS), test_validate_data);
        csvwrite(sprintf('train_segbins%d.csv',INTENSITY_BINS), training_data);
    end
else
    if (FCC)
        csvwrite(sprintf('test_validate_bins%d_fcc.csv',INTENSITY_BINS), test_validate_data);
        csvwrite(sprintf('train_bins%d_fcc.csv',INTENSITY_BINS), training_data);
    else
        csvwrite(sprintf('test_validate_bins%d.csv',INTENSITY_BINS), test_validate_data);
        csvwrite(sprintf('train_bins%d.csv',INTENSITY_BINS), training_data);
   end
end

end %endfor
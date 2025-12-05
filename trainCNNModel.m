%% 1. Load and Prepare the Data Set ---

spectrogramDir = 'vera_canan_kaggle_astjan_spec/';
imageSize = [64, 64, 1]; 

% Load images using ImageDatastore.
imds = imageDatastore(spectrogramDir, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames'); 
numClasses = numel(categories(imds.Labels));

% Split the Data Set
rng(42); % For reproducibility ?
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomize'); % 90% train set 10% test set

% Set ReadFcn for normalization (to the 0-1 range using im2double)
imdsTrain.ReadFcn = @(filename) im2double(imread(filename));
imdsTest.ReadFcn  = @(filename) im2double(imread(filename));


% Data Augmentation
pixelRange = [-5 5]; 
scaleRange = [0.9 1.1]; 
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandXTranslation', pixelRange, ...
    'RandYTranslation', pixelRange, ...
    'RandXScale', scaleRange, ...
    'RandYScale', scaleRange);

augimdsTrain = augmentedImageDatastore(imageSize, imdsTrain, ...
    'DataAugmentation', imageAugmenter); 


%% 2. CNN Model (GÜNCELLENMİŞ VERSİYON)
imageSize = [64 64 1];
layers = [
    imageInputLayer(imageSize, 'Name', 'input')

    % CONV-BN-RELU-POOL Block 1: (64x64 -> 32x32)
    convolution2dLayer([3 3], 32, 'Padding', 'same', 'Name', 'conv1') 
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer([2 2], 'Stride', [2 2], 'Name', 'maxpool1') 

    % CONV-BN-RELU-POOL Block 2: (32x32 -> 16x16)
    convolution2dLayer([3 3], 64, 'Padding', 'same', 'Name', 'conv2') 
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer([2 2], 'Stride', [2 2], 'Name', 'maxpool2') 
    
    % CONV-BN-RELU Block 3: (16x16 -> 8x8)
    convolution2dLayer([3 3], 128, 'Padding', 'same', 'Name', 'conv3') 
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')
    maxPooling2dLayer([2 2], 'Stride', [2 2], 'Name', 'maxpool3') % 8x8
    
    dropoutLayer(0.4, 'Name', 'dropout_conv') % Dropout oranını artırdık
    
    % Fully Connected Layers (Basitleştirilmiş)
    fullyConnectedLayer(128, 'Name', 'fc1') 
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    dropoutLayer(0.5, 'Name', 'dropout_fc') % Ezberlemeyi önlemek için daha agresif dropout
    
    % Çıkış Katmanı
    fullyConnectedLayer(10, 'Name', 'fcOut') 
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];
%% 3. Set Training Parameters and Train

% Used 'adam' optimization
options = trainingOptions('adam', ... 
    'InitialLearnRate', 0.001, ...       
    'MaxEpochs', 100, ...               
    'MiniBatchSize', 32, ...            
    'ValidationData', imdsTest, ...     
    'ValidationFrequency', 30, ...      
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'ValidationPatience', 40); % Aşırı öğrenmeyi önlemek için Early Stopping eklendi

fprintf('Model is training (Epochs: 100, Batch Size:32, ValidationPatience: 30)...\n');
net = trainNetwork(augimdsTrain, layers, options);
fprintf('Training completed.\n');

%% 4. Evaluate the Model 
save('digit_09_model_v4_vcka.mat', 'net', 'imageSize'); 
fprintf('Model saved as "digit_09_model_v4_vcka.mat" .\n');

% Calculate accuracy on the test data
YPred = classify(net, imdsTest);
accuracy = sum(YPred == imdsTest.Labels) / numel(imdsTest.Labels);
fprintf('Test Accuracy: %.4f\n', accuracy);
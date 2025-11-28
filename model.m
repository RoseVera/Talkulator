%% CNN Training on Mel-Spectrogram Dataset
clear; clc;

%% --------- Load Dataset ---------
inputFolder = "mel_spectrogram_dataset";

% Load images with labels from folder names
imds = imageDatastore(inputFolder, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

% Split into training (80%) and test (20%) sets
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

% Resize images to 64x64 (CNN requires fixed input size)
inputSize = [64 64];

% Resize and convert to grayscale
imdsTrain.ReadFcn = @(filename)imresize(rgb2gray(imread(filename)), [64 64]);
imdsTest.ReadFcn  = @(filename)imresize(rgb2gray(imread(filename)), [64 64]);


%% --------- Define CNN ---------
numClasses = numel(categories(imdsTrain.Labels));

layers = [
    imageInputLayer([64 64 1])    % Grayscale images

    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)
 
    convolution2dLayer(3,64,'Padding','same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2,'Stride',2)

    fullyConnectedLayer(128)
    reluLayer
    dropoutLayer(0.5)

    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

%% --------- Training Options ---------
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', imdsTest, ...
    'ValidationFrequency', 30, ...
    'Verbose', false, ...
    'Plots','training-progress');

%% --------- Train CNN ---------
net = trainNetwork(imdsTrain, layers, options);

%% --------- Evaluate Model ---------
YPred = classify(net, imdsTest);
YTest = imdsTest.Labels;

accuracy = sum(YPred == YTest)/numel(YTest);
disp("Test Accuracy: " + accuracy*100 + "%");

% Confusion Matrix
figure;
confusionchart(YTest, YPred);
title('Confusion Matrix for Digit Classification');

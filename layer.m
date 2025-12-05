%% 2. CNN Model 

imageSize = [64 64 1];
layers = [
    imageInputLayer(imageSize, 'Name', 'input') % Input layer (64x64x1 grayscale image)

    % CONV-BN-RELU Block 1:
    convolution2dLayer([2 2], 32, 'Name', 'conv1') 
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')

    % CONV-BN-RELU Blok 2: 
    convolution2dLayer([2 2], 48, 'Name', 'conv2')
    batchNormalizationLayer('Name', 'bn2')
    reluLayer('Name', 'relu2')

    % CONV-BN-RELU Blok 3: 
    convolution2dLayer([2 2], 120, 'Name', 'conv3')
    batchNormalizationLayer('Name', 'bn3')
    reluLayer('Name', 'relu3')

    % Max Pooling and Dropout
    maxPooling2dLayer([2 2], 'Stride', [2 2], 'Name', 'maxpool')
    dropoutLayer(0.25, 'Name', 'dropout1')

    % Fully Connected Layers
    fullyConnectedLayer(128, 'Name', 'fc1') 
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    dropoutLayer(0.25, 'Name', 'dropout2')

    fullyConnectedLayer(64, 'Name', 'fc2')
    batchNormalizationLayer('Name', 'bn5')
    reluLayer('Name', 'relu5')
    dropoutLayer(0.4, 'Name', 'dropout3')

    fullyConnectedLayer(10, 'Name', 'fcOut') % Output layer for 10 classes ( 0-9)
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];
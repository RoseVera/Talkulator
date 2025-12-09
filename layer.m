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

%%for 360 sample digit dataset
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
    
    dropoutLayer(0.3, 'Name', 'dropout_conv') 
    
    % Fully Connected Layers
    fullyConnectedLayer(128, 'Name', 'fc1') 
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    dropoutLayer(0.4, 'Name', 'dropout_fc') % more aggressive dropout to prevent overfitting
    
    % Output Layer
    fullyConnectedLayer(10, 'Name', 'fcOut') 
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];
%% 3. Set Training Parameters and Train

% Used 'adam' optimization
options = trainingOptions('adam', ... 
    'InitialLearnRate', 0.001, ...       
    'MaxEpochs', 100, ...               
    'MiniBatchSize', 64, ...            
    'ValidationData', imdsTest, ...     
    'ValidationFrequency', 30, ...      
    'Shuffle', 'every-epoch', ...
    'Plots', 'training-progress', ...
    'Verbose', false, ...
    'ValidationPatience', 20); % Early Stopping to prevent overfitting



%%for operators 5 class
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
    
    dropoutLayer(0.4, 'Name', 'dropout_conv') 
    
    % Fully Connected Layers
    fullyConnectedLayer(128, 'Name', 'fc1') 
    batchNormalizationLayer('Name', 'bn4')
    reluLayer('Name', 'relu4')
    dropoutLayer(0.5, 'Name', 'dropout_fc') % more aggressive dropout to prevent overfitting
    
    % Output Layer
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
    'ValidationPatience', 20); % Early Stopping to prevent overfitting


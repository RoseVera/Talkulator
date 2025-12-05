%% Speech Recognition Calculator (Main Program)
clear; clc;

%% 1. Load Model and Settings ---
try
    % Load your trained model 
    load('digit_09_model_v4_vcka.mat', 'net');
    disp('Model main_model.mat loaded successfully.');
catch
    error('ERROR: main_model.mat file not found or could not be loaded. Please check the path.');
end

% Recording Settings (Must be identical to Training Settings)
Fs = 16000;
duration = 1.0;
recObj = audiorecorder(Fs, 16, 1);
spectrogramDimensions = [64, 64]; 
numMelBands = spectrogramDimensions(1);
targetTimeLength = spectrogramDimensions(2);
winSize = round(0.025 * Fs); 
hopSize = round(0.010 * Fs); 
nfft = 512;

% Global Variables
expressionBuffer = {};   % Cell array to store the predicted mathematical symbols (e.g., '5', '+')
predictionHistory = {};  % Not strictly used for math, but kept for historical tracking

disp('====================================================');
disp('   SPEECH RECOGNITION CALCULATOR (TERMINAL)');
disp('   Speak Digits (0-9) and Operators (+, -, *, /, =).');
disp('   Say "equals" to finish the calculation.');
disp('====================================================');

%% 2. Main Processing Loop 
while true
    if strcmpi(input_str, 'q')
        break;
    end
    fprintf('\nCurrent Expression: %s', strjoin(expressionBuffer, ' '));
    input('\nPress ENTER and prepare to speak, type "q" and press ENTER to quit...', 's');

    % Record Audio
    disp('️🎙RECORDING STARTED: Speak now...');
    recordblocking(recObj, duration);
    disp('🛑 RECORDING FINISHED.');
    y = getaudiodata(recObj);

     %Normalize signal power
    max_amplitude = max(abs(y));
    if max_amplitude > 0.01 
        y = y / max_amplitude; 
    end
   
    %% 3. Audio Pre-processing and Spectrogram Creation 
    % Spectrogram (MelSpectrogram)
    [S, ~, ~] = melSpectrogram(y, Fs, ...
        'Window', hamming(winSize, 'periodic'), ...
        'OverlapLength', winSize - hopSize, ...
        'FFTLength', nfft, ...
        'NumBands', numMelBands, ...
        'FrequencyRange', [0, Fs/2]);
    
    % Convert to dB Scale
    S_dB = 20*log10(abs(S) + 1e-6); 
    
    % Resize the Time Dimension (To obtain the 64x64 dimension)
    S_resized = imresize(S_dB, [numMelBands, targetTimeLength], 'nearest');
    
    % Normalization and Color Inversion (Same as generateSpectrograms.m)
    S_normalized = S_resized - min(S_resized(:));
    S_normalized = S_normalized / max(S_normalized(:));
    imgData_double = 1 - S_normalized; % Invert colors, resulting in double (0-1) type
    imgForCNN = reshape(imgData_double, [spectrogramDimensions, 1]); 

    % Predict
    YPred = classify(net, imgForCNN);
    predictedLabel = string(YPred);
    disp(predictedLabel);

    % Map the predicted word label to its mathematical symbol
    symbol = mapPredictionToSymbol(predictedLabel);

    fprintf('\n-> PREDICTION RESULT: %s (Symbol: %s)\n', predictedLabel, symbol);
    
    %% 3. Correction Mechanism (Press R to redo) 
    while true
        action = input('Press ENTER if prediction is correct, or (R) to record again: ', 's');
        
        if isempty(action) % User pressed ENTER, prediction is accepted
            expressionBuffer{end+1} = symbol;
            predictionHistory{end+1} = predictedLabel;
            break; 

        elseif strcmpi(action, 'R') % User pressed R, wants to re-record
            if ~isempty(expressionBuffer)
                % Remove the last accepted symbol from the buffer (undo)
                expressionBuffer(end) = []; 
                predictionHistory(end) = [];
                fprintf('<- LAST ENTRY CANCELLED. Current Expression: %s\n', strjoin(expressionBuffer, ' '));
            else
                disp('Expression is empty. No entry to undo.');
            end
            % Break out of the correction loop to return to the main recording loop
            break; 
        else
            disp('Invalid input. Please press ENTER or R.');
        end
    end
    
    %% 4. Calculation and Exit Control 
    if strcmp(symbol, '=')
        fprintf('\n====================================');
        fprintf('\nCALCULATION TERMINATING...');
        
        % Remove the trailing '=' sign from the buffer
        if strcmp(expressionBuffer{end}, '=')
            expressionBuffer(end) = []; 
        end
        
        % Join the expression elements and evaluate
        finalExpression = strjoin(expressionBuffer, '');
        
        try
            % Use MATLAB's 'eval' function to solve the mathematical expression
            result = eval(finalExpression);
            fprintf('\nOPERATION: %s\n', finalExpression);
            fprintf('RESULT: %g\n', result);
        catch
            fprintf('\nERROR: An invalid mathematical expression occurred. Expression: %s\n', finalExpression);
            disp('Example Errors: "5 + +" or "3 4"');
        end
        
        disp('====================================');
        break; % Exit the main loop
    end
end
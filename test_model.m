%% ============================================
%  LIVE SPEECH RECOGNITION USING TRAINED MODEL
%  Converts spoken digit → model prediction
% =============================================

clear; clc;

%% --- Load Trained Model ---
load("trained_digit_cnn.mat", "net");   % your trained CNN model

%% --- Recording Settings ---
fs = 16000;         % same as training
duration = 1;       % 1 second recording
recObj = audiorecorder(fs, 16, 1);

disp("======================================");
disp("   SPEAK THE DIGIT OR OPERATOR");
disp("   Press ENTER to start recording...");
disp("======================================");
pause;

disp("Recording...");
recordblocking(recObj, duration);
disp("Done!");

audioData = getaudiodata(recObj);

%% --- Preprocess Audio ---
% Resample (just in case)
audioData = audioData(:,1);
audioData = resample(audioData, 16000, fs);
fs = 16000;

%% --- Generate mel spectrogram exactly like training ---
melSpec = melSpectrogram(audioData, fs, ...
    "Window", hamming(512, "periodic"), ...
    "OverlapLength", 256, ...
    "FFTLength", 512, ...
    "NumBands", 64, ...
    "ApplyLog", true);

% Convert to RGB image using imagesc + jet colormap
fig = figure('Visible','off');
imagesc(melSpec);
axis off;
colormap jet;

frame = getframe(gca);
img = frame.cdata;

close(fig);

%% === Save spectrogram image (OPTIONAL) ===
timestamp = datestr(now,'yyyymmdd_HHMMSS');
outputFile = sprintf("live_melspec_%s.png", timestamp);

imwrite(img, outputFile);
fprintf("Spectrogram saved as: %s\n", outputFile);

% Resize to 64×64 grayscale (same preprocessing as imds)
img = imresize(rgb2gray(img), [64 64]);

img = reshape(img, [64 64 1]);
%% --- Predict ---
YPred = classify(net, img);

disp("======================================");
fprintf("You said:  %s\n", string(YPred));
disp("======================================");

%% Convert to integer if digit
if isstrprop(string(YPred), 'digit')
    predictedNumber = str2double(string(YPred));
    fprintf("Predicted integer: %d\n", predictedNumber);
else
    fprintf("Operator detected: %s\n" + ...
        "", string(YPred));
end

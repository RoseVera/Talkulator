%% MEL-SPECTROGRAM FEATURE EXTRACTION FOR CNN
clear; clc;

inputFolder = "digit_dataset";              
outputFolder = "mel_spectrogram_dataset"; 
mkdir(outputFolder);

categories = dir(inputFolder);
categories = categories([categories.isdir] & ~startsWith({categories.name}, "."));

for c = 1:length(categories)
    label = categories(c).name;
    disp("Processing: " + label)

    inDir = fullfile(inputFolder, label);
    outDir = fullfile(outputFolder, label);
    mkdir(outDir);

    audioFiles = dir(fullfile(inDir, "*.wav"));

    for i = 1:length(audioFiles)
        filePath = fullfile(audioFiles(i).folder, audioFiles(i).name);

        % -------- LOAD AUDIO --------
        [y, fs] = audioread(filePath);

        y = y(:,1);                    
        y = resample(y, 16000, fs);    
        fs = 16000;

        % -------- MEL SPECTROGRAM (Correct syntax) --------
        melSpec = melSpectrogram(y, fs, ...
            "Window", hamming(512, "periodic"), ...
            "OverlapLength", 256, ...
            "FFTLength", 512, ...
            "NumBands", 64, ...
            "ApplyLog", true);

        % -------- SAVE AS IMAGE --------
        fig = figure('Visible', 'off');
        imagesc(melSpec);
        axis off;
        colormap jet;

        outFile = fullfile(outDir, replace(audioFiles(i).name, ".wav", ".png"));
        saveas(fig, outFile);

        close(fig);
    end
end

disp("Mel-spectrogram dataset creation complete!");

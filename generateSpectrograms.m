function generateSpectrograms(audioDir, spectrogramDir, spectrogramDimensions)
% generateSpectrograms: Converts audio files into Mel-spectrogram images
% and preserves the directory structure.
%
% audioDir: The main directory containing audio files (e.g., 'recordings/').
% spectrogramDir: The main directory where spectrograms will be saved (e.g., 'spectrograms/').
% spectrogramDimensions: The dimensions of the spectrogram image [Height, Width]. Defaults to [64, 64].

    if nargin < 3
        spectrogramDimensions = [64, 64]; % [Height, Width]
    end

    % Create the output root directory
    if ~exist(spectrogramDir, 'dir')
        mkdir(spectrogramDir);
    end

    % Find the Label/Digit Subdirectories (0 through 9)
    subDirs = dir(audioDir);
    subDirs = subDirs([subDirs.isdir]);% Get only directories
    subDirs = subDirs(~ismember({subDirs.name},{'.','..'})); % Exclude '.' and '..'
    
    numClasses = length(subDirs);

    for i = 1:numClasses
        className = subDirs(i).name;
        currentAudioPath = fullfile(audioDir, className);
        currentSpectrogramPath = fullfile(spectrogramDir, className); % Preserve the subfolder structure
        
        if ~exist(currentSpectrogramPath, 'dir')
            mkdir(currentSpectrogramPath);
        end

        filePattern = fullfile(currentAudioPath, '*.wav');
        theFiles = dir(filePattern);
        
        numFiles = length(theFiles);
        fprintf('Sınıf %s için %d dosya işleniyor...\n', className, numFiles);

        for k = 1:numFiles
            baseFileName = theFiles(k).name;
            fullAudioPath = fullfile(currentAudioPath, baseFileName);
            
           % Read the audio file
            [y, Fs] = audioread(fullAudioPath);
            
           % Convert to single channel (if stereo)
            if size(y, 2) > 1
                y = mean(y, 2);
            end
            
            %%--- Mel-Spectrogram Generation (Consistent Parameters) -----
            winSize = round(0.025 * Fs); % 25 ms window size
            hopSize = round(0.010 * Fs); % 10 ms hop size (15 ms overlap)
            nfft = 512;% FFT length
            numMelBands = spectrogramDimensions(1);
            
            [S, F, T] = melSpectrogram(y, Fs, ...
                'Window', hamming(winSize, 'periodic'), ...
                'OverlapLength', winSize - hopSize, ...
                'FFTLength', nfft, ...
                'NumBands', numMelBands, ...
                'FrequencyRange', [0, Fs/2]);
            
            S_dB = 20*log10(abs(S) + 1e-6); % Convert magnitude to dB scale
            targetTimeLength = spectrogramDimensions(2);

            % Resize the time axis to match the target width (64)
            S_resized = imresize(S_dB, [numMelBands, targetTimeLength], 'nearest');
            
            % Normalize the image to 0-1 range and convert to inverted grayscale ('gray_r')
            S_normalized = S_resized - min(S_resized(:));
            S_normalized = S_normalized / max(S_normalized(:));
            imgData_double = 1 - S_normalized; % Invert colors
            imgData_uint8 = uint8(imgData_double * 255);
            
            % Save as PNG
            outputFileName = replace(baseFileName, '.wav', '.png');
            fullOutputPath = fullfile(currentSpectrogramPath, outputFileName);
            % Save the image data using the grayscale colormap (256 levels)
            imwrite(imgData_uint8, gray(256), fullOutputPath, 'png'); 
        end
    end
    
    disp('All spectrograms have been successfully created and directory structure preserved.');
end
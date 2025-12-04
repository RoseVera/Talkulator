% live_digit_recognition.m
% Digit recognition using the trained CNN model and microphone input
% --- 1. Load Model and Parameters ---
modelName = 'digit_09_model_v3.mat';

if ~exist(modelName, 'file')
    error(['Trained model file not found: ', modelName, '. Please run trainCNNModel.m first.']);
end
load(modelName, 'net', 'imageSize'); 
disp(['Trained model "', modelName, '" loaded successfully.']);

% --- 2. Fixed Spectrogram Parameters (Consistent with generateSpectrograms.m) ---
% CRITICAL FOR LIVE TEST: These parameters must be identical to the steps used for training data creation.
Fs = 16000; % Sample rate for live recording (Must match training data Fs. 16000 Hz is commonly used.)
recordDuration = 1; % Recording duration for each spoken digit (seconds)
spectrogramDimensions = [64, 64]; 
numMelBands = spectrogramDimensions(1);
targetTimeLength = spectrogramDimensions(2);
winSize = round(0.025 * Fs); 
hopSize = round(0.010 * Fs); 
nfft = 512;

% --- YENİ EKLENEN KAYDETME DEĞİŞKENLERİ ---
audioDir_LiveTest = 'live_recordings/';
spectrogramDir_LiveTest = 'live_spectrograms/';
if ~exist(audioDir_LiveTest, 'dir'), mkdir(audioDir_LiveTest); end
if ~exist(spectrogramDir_LiveTest, 'dir'), mkdir(spectrogramDir_LiveTest); end
kayitSayaci = 0; % Benzersiz dosya adı oluşturmak için sayaç
% ----------------------------------------

% Initialize the Audio Recorder
recObj = audiorecorder(Fs, 16, 1); % 16-bit, mono channel

disp('--------------------------------------------------');
disp('LIVE DIGIT RECOGNITION STARTED');
disp(['Recording Duration: ', num2str(recordDuration), ' seconds']);
disp(['Sample Rate (Fs): ', num2str(Fs), ' Hz']);
disp(['Kayıtlar buraya kaydedilecek: ', audioDir_LiveTest, ' ve ', spectrogramDir_LiveTest]);
disp('--------------------------------------------------');
disp('Ready. Press ENTER to speak a digit, type "q" and press ENTER to quit.');

while true
    input_str = input('', 's');
    if strcmpi(input_str, 'q')
        break;
    end
    
    kayitSayaci = kayitSayaci + 1;
    baseFileName = ['test_', num2str(kayitSayaci, '%03d')];
    
    disp('🎙️ Recording... Speak now.');
    recordblocking(recObj, recordDuration); % Start and wait for recording
    disp('🛑 Recording Completed.');
    
    y = getaudiodata(recObj); % Get audio data

    % --- YENİ KOD: Sinyal Gücünü Normalleştirme ---
    max_amplitude = max(abs(y));
    if max_amplitude > 0.01 % Çok sessiz değilse
        % Maksimum genliği 1'e normalleştirerek sesi yükselt.
        y = y / max_amplitude; 
    end
    
    % --- Ses Dosyasını Kaydetme (YENİ) ---
    fullAudioPath = fullfile(audioDir_LiveTest, [baseFileName, '.wav']);
    audiowrite(fullAudioPath, y, Fs);
    % --------------------------------------
    
    % --- 3. Audio Pre-processing and Spectrogram Creation ---
    
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
    
    % --- Spektrogram Dosyasını Kaydetme (YENİ) ---
    fullSpectrogramPath = fullfile(spectrogramDir_LiveTest, [baseFileName, '.png']);
    imgData_uint8_to_save = uint8(imgData_double * 255); % PNG'ye kaydederken uint8 kullanmalıyız
    imwrite(imgData_uint8_to_save, gray(256), fullSpectrogramPath, 'png'); 
    % ----------------------------------------------
    
    % Prepare for CNN Input (CNN DOUBLE beklediği için tekrar double tipini kullanıyoruz)
    imgForCNN = reshape(imgData_double, [spectrogramDimensions, 1]); 
    
    % --- 4. Classification and Display Result ---
    YPred = classify(net, imgForCNN);
    
    predictedDigit = char(YPred);
    
    disp('✨ Predicted Digit:');
    disp(['🎉 ', predictedDigit]);
    disp(['[Dosya Adı: ', baseFileName, '.wav / .png]']);
    disp('--------------------------------------------------');
    disp('Press ENTER for a new recording, type "q" and press ENTER to quit.');
end

disp('Live digit recognition terminated.');
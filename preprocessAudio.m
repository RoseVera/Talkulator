function y = preprocessAudio(y, fs)
% PREPROCESSAUDIO: Applies standard pre-processing steps to an audio signal
% for speech recognition.
%
% y: Input audio data vector.
% fs: Sample rate of the input audio.
%
% Output y: Processed audio vector (16000 samples, 16000 Hz).

    %% 1. Mono (only one chanel)
    if size(y, 2) > 1
        y = y(:,1);
    end

    %% 2. Resample to 16 kHz 
    targetFs = 16000;
    if fs ~= targetFs
        y = resample(y, targetFs, fs);
    end
    fs = targetFs;

    %% 3. Amplitude Normalization 
    max_amplitude = max(abs(y));
    if max_amplitude > 1e-6 
        y = y / max_amplitude; 
    end
    
    % 4. Pre-emphasis
    % y = filter([1 -0.97], 1, y);
    
    % 5. Remove silence
    threshold = 0.05; 
    idx = find(abs(y) > threshold);
    if ~isempty(idx)
        y = y(idx(1):idx(end)); 
    end

    % 6. Fix length 
    targetLen = 16000;

    if length(y) < targetLen
        y(end:targetLen) = 0;   
    else
        y = y(1:targetLen);     
    end
end
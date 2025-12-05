function y = preprocessAudio(y, fs)
% PREPROCESSAUDIO: Applies standard pre-processing steps to an audio signal
% for speech recognition.
%
% y: Input audio data vector.
% fs: Sample rate of the input audio.
%
% Output y: Processed audio vector (16000 samples, 16000 Hz).

    % 1. Mono (Sadece tek kanal kullan)
    if size(y, 2) > 1
        y = y(:,1);
    end

    % 2. Resample to 16 kHz (Örnekleme Hızını Sabitle)
    targetFs = 16000;
    if fs ~= targetFs
        y = resample(y, targetFs, fs);
    end
    fs = targetFs; % fs değerini güncelle

    % 3. Amplitude Normalization (Genlik Normalizasyonu)
    % Sesin çok sessiz olması durumunda genliği 1'e normalleştir.
    max_amplitude = max(abs(y));
    if max_amplitude > 1e-6 % Çok sessiz değilse (Sıfıra bölme hatasını önlemek için 1e-6)
        y = y / max_amplitude; 
    end
    
    % --- [Daha Önce Kullanılan Ama Bu Projede Yorum Satırı Olan Adımlar] ---
    % 4. Pre-emphasis (Ön Vurgu - Gerekli ise etkinleştirin)
    % y = filter([1 -0.97], 1, y);
    
    % 5. Remove silence (Sessizlik Kaldırma - Veri çeşitliliğiniz azsa bu zorunludur)
    % Bu adım, spektrogramın sadece konuşulan sesi içermesi için kritik olabilir.
    threshold = 0.05; 
    idx = find(abs(y) > threshold);
    if ~isempty(idx)
        y = y(idx(1):idx(end)); % Konuşmanın başladığı yerden bittiği yere kadar kes
    end

    % 6. Fix length (Uzunluğu Sabitle) -> 1 second (16000 samples)
    targetLen = 16000;

    if length(y) < targetLen
        % Ses kısa ise sıfır doldurma (padding)
        y(end:targetLen) = 0;   
    else
        % Ses uzun ise kesme (cutting)
        y = y(1:targetLen);     
    end
end
function filtsignal = PreprocessEEG(temp, rawsignal)

% 0. Reference the raw EEG signals with Fz (I am assuming that this is already done for kEEG NIRS)
% 1. Bandpass filter the data to remove frequencies greater than 50 Hz and lower than 1Hz
filtsignal = bandpass_butterworth(double(rawsignal), temp.filt_bandpass, temp.Fs, temp.order);



function filtered_signal = bandpass_butterworth(inputsignal, cutoff_freqs, Fs, order)

nyquist_freq = Fs/2;            % Nyquist frequency

Wn = cutoff_freqs/nyquist_freq;    % Non-dimensional frequency

[filtb, filta] = butter(order, Wn, 'bandpass'); % construct the filter

filtered_signal = filtfilt(filtb, filta, inputsignal); % filter the data with zero phase
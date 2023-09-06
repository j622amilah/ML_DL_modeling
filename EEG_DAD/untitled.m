close all
clear all
clc

load dad

temp.data_eegch = dad.data_eegchall{dad.popup_splocValue,1};

[a1, a2] = size(dad.database);    
for EXPs = 1:a2
    DADmethod.database{EXPs,1} = dad.database(:,EXPs);
end
m
DADmethod.NumOfEXPs = a2+1; % Total number of datasets to compare at one time
DADmethod.NumOfCHs = 1;     % Loading 1 spatial channel at a time

% Add the new dataset at the end
DADmethod.database{DADmethod.NumOfEXPs, 1} = temp.data_eegch;

statsDAD.order = 4;
statsDAD.filt_bandpass = [1 50];    % Low and high frequencies for bandpass filter
statsDAD.Fs = dad.data.samplingrate;	 % 256


for i = 1:DADmethod.NumOfEXPs
    DADmethod.filtsignal{i,1} = PreprocessEEG(statsDAD, DADmethod.database{i,1});
end

DADmethod.vec_Acc_STEPlast = DADmethod.filtsignal;

dad.NumOfsampledCHs = DADmethod.NumOfEXPs*DADmethod.NumOfCHs;


%%


% (STEP e1) - Epoch each channel (cut the data in each channel every 10 seconds)
DADmethod.EpochLength = 10;
DADmethod.EpochStart = 1;
DADmethod.EpochEnd = (statsDAD.Fs*DADmethod.EpochLength) + 1;
j = 1;

for i = 1:dad.NumOfsampledCHs
    while ( length(DADmethod.vec_Acc_STEPlast{i,1}) >= DADmethod.EpochEnd )
        DADmethod.(sprintf('vec_epoch_Nsamp%d_STEPe1', i)){j,1} = DADmethod.vec_Acc_STEPlast{i,1}(DADmethod.EpochStart:DADmethod.EpochEnd);
        DADmethod.EpochStart = 1 + DADmethod.EpochEnd;
        DADmethod.EpochEnd = (statsDAD.Fs*DADmethod.EpochLength) + 1 + DADmethod.EpochEnd;
        j = j + 1;
    end

    DADmethod.EpochStart = 1;
    DADmethod.EpochEnd = (statsDAD.Fs*DADmethod.EpochLength) + 1;
    j = 1;
end

for i = 1:dad.NumOfsampledCHs
    statsDAD.rej_struct{i,1} = zeros(length(DADmethod.(sprintf('vec_epoch_Nsamp%d_STEPe1', i))), 1);
    statsDAD.rej_struct_gamma{i,1} = zeros(length(DADmethod.(sprintf('vec_epoch_Nsamp%d_STEPe1', i))), 1);
end


%%


% (STEP e2) - Baseline correct - zero mean the data and append "good" datasets to the epochs
for i = 1:dad.NumOfsampledCHs
    for j = 1:length(DADmethod.(sprintf('vec_epoch_Nsamp%d_STEPe1', i)))
        if mean(DADmethod.(sprintf('vec_epoch_Nsamp%d_STEPe1', i)){j,1}) > 0
            DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_STEPe2', i)){j,1} = ...
                (DADmethod.(sprintf('vec_epoch_Nsamp%d_STEPe1', i)){j,1} - mean(DADmethod.(sprintf('vec_epoch_Nsamp%d_STEPe1', i)){j,1}));
        else
            DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_STEPe2', i)){j,1} = ...
                (mean(DADmethod.(sprintf('vec_epoch_Nsamp%d_STEPe1', i)){j,1}) - DADmethod.(sprintf('vec_epoch_Nsamp%d_STEPe1', i)){j,1});
        end
    end
end



% Create a sinusodial wave containing alpha, beta frequencies, to use as an absolute measure (baseline for signal-to-noise)
fs = 256;	% sampling frequency
dur = 10;    % duration in seconds
t = (1/fs):(1/fs):dur;  %time array
freq = 8;    % frequency components (Hz)
amp = 0.2;          % db = 80;    % intensity, 10^((db-100)/20); % amplitude
baseline = amp*sin(2*pi*freq*t);
baselineSIG = [0, baseline]';
% figure; plot(baselineSIG)


for i = 1:dad.NumOfsampledCHs
    for j = 1:length(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_STEPe2', i)))
        temp2.x{i,1}{j,1} = DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_STEPe2', i)){j,1};  % epoch I want to test
        
        % --------
        % Randomly select one of the good data sets from the database
        % b = 1;
        % c = dad.NumOfsampledCHs-1;
        % r = ceil(b + (c-b).*rand(1,1));
        % temp2.y{i,1}{j,1} = DADmethod.database{r,1};  % the "good" epoch length signal
        % temp2.y_fft{i,1}{j,1} = fft(temp2.y{i,1}{j,1}, N);  % Fourier transform
        % temp2.y_fft_gain{i,1}{j,1} = abs(temp2.y_fft{i,1}{j,1}(1:round(length(f)/2)));  % gain (frequency response)
        % --------
        
        N = length(temp2.x{i,1}{j,1});
        f = (0:N-1)*statsDAD.Fs/(N-1);   % corresponding frequency vector

        temp2.x_fft{i,1}{j,1} = fft(temp2.x{i,1}{j,1}, N);  % Fourier transform
        temp2.x_fft_gain{i,1}{j,1} = abs(temp2.x_fft{i,1}{j,1}(1:round(length(f)/2)));  % gain (frequency response)
        
        temp2.baselineSIG_fft = fft(baselineSIG, N);  % Fourier transform
        temp2.baselineSIG_fft_gain = abs(baselineSIG_fft(1:round(length(f)/2)));  % gain (frequency response)
        

        temp2.epoch_snr{i,1}(j,1) = (20*log10(norm(temp2.x_fft_gain{i,1}{j,1})./...
            norm(temp2.x_fft_gain{i,1}{j,1} - temp2.baselineSIG_fft_gain)));
    end
end

for i = 1:dad.NumOfsampledCHs
    temp2.epoch_snr_zscore{i,1} = zscore(temp2.epoch_snr{i,1}); % Matlab's zscore: z = (data point-pop_mean)/pop_std
end

%%

% --------------------------
% If an epoch is rejected, make the epoch = zero but hold the same data length, 
% by retaining the same data length we can count the correct epochs
% --------------------------
% Checker Program
q = 1;
for i = 1:dad.NumOfsampledCHs
    for j = 1:length(temp2.epoch_snr_zscore{i,1})
        if abs(temp2.epoch_snr_zscore{i,1}(j,1)) > 2 % An epoch whose amplitude range has a z-score of +/-2 would be deemed contaminated
            DADmethod.Rej_snr_epoch(i,1) = q;
            q = q + 1;
        else
            DADmethod.Rej_snr_epoch(i,1) = 0;
        end
    end
    q = 1;
end

statsDAD.Rej_snr_epoch = DADmethod.Rej_snr_epoch;

% 60% good and 40% test, 150*(0.6)= 90 cells from good and 60 cells from test
Length_of_cell = 150;
gRatio = 0.6;
tRatio = 0.4;
testCellsAmt = 60;

 % --------------------------
% Mixture of subjects with High, Moderate, and Low SNR
% size is 512*5 + 1 = [2561 1], 2 second chunks from each column in the database
% --------------------------
for i = 1:Length_of_cell*(gRatio)    % number of cells needed to append = 90 cells

    c = dad.NumOfsampledCHs-1;
    r1 = ceil(1 + (c-1).*rand(1,1));
    r2 = ceil(1 + (c-1).*rand(1,1));
    r3 = ceil(1 + (c-1).*rand(1,1));
    r4 = ceil(1 + (c-1).*rand(1,1));
    r5 = ceil(1 + (c-1).*rand(1,1));

    b = 1;
    d = 512;
    s1 = ceil(b + (d-b).*rand(1,1));
    e1 = s1+511;
    e11 = s1+512;

%     DADmethod.pre_goodepochSet{i,1} = [DADmethod.database{r1,1}(s1:e1,1); DADmethod.database{r2,1}(s1:e1,1); ...
%         DADmethod.database{r3,1}(s1:e1,1); DADmethod.database{r4,1}(s1:e1,1); DADmethod.database{r5,1}(s1:e11,1)];

%     DADmethod.pre_goodepochSet{i,1} = [DADmethod.database{1,1}(s1:e1,1); DADmethod.database{1,1}(s1:e1,1); ...
%         DADmethod.database{1,1}(s1:e1,1); DADmethod.database{1,1}(s1:e1,1); DADmethod.database{1,1}(s1:e11,1)];

    DADmethod.pre_goodepochSet{i,1} = [DADmethod.database{1,1}(s1:e1,1); DADmethod.database{2,1}(s1:e1,1); ...
        DADmethod.database{1,1}(s1:e1,1); DADmethod.database{2,1}(s1:e1,1); DADmethod.database{1,1}(s1:e11,1)];
end

for i = 1:dad.NumOfsampledCHs
    o1 = Length_of_cell*(gRatio);  % 90 cells
    o2 = Length_of_cell - o1;   % 60 cells
    DADmethod.times2div(i,1) = ceil(length(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_STEPe2', i)))/o2);

    o1_start = 1;
    o1_end = o1;
    o2_start = 1;
    o2_end = o2;

    for ii = 1:DADmethod.times2div(i,1)
        if ii == DADmethod.times2div(i,1)

            if ii == 1
                if length(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_STEPe2', i))) < (o2_end-1)    % For really short data sets 
                    o2 = length(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_STEPe2', i)));  % 60
                    o1 = floor(o2/(tRatio));  % 60/(0.4) = 150 cells

                    DADmethod.Length_of_cell{i,1}(ii,1) = o1 + o2;   % 91 cells
                    DADmethod.Length_of_testcell{i,1}(ii,1) = o2;    % 26 cells

                    o1_end = o1;
                    o2_end = o2;
                end
            else

                % This is the last loop, so o1 and o2 need to be adjusted because the total length is less than 150
                o2 = length(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_STEPe2', i))) - o2*(DADmethod.times2div(i,1)-1);     % reminder - 60 cells
                o1 = floor(o2/(tRatio));  % 17/(0.4) = 150 cells

                % Exception: if the last cell evenly divides
                if o1 > Length_of_cell*(gRatio) %90
                    o1 = Length_of_cell*(gRatio);  % 90 cells

                    DADmethod.Length_of_cell{i,1}(ii,1) = Length_of_cell;   % 150 cells
                    DADmethod.Length_of_testcell{i,1}(ii,1) = o2;
                else
                    DADmethod.Length_of_cell{i,1}(ii,1) = o1 + o2;   % less than 150 cells
                    DADmethod.Length_of_testcell{i,1}(ii,1) = o2;
                end

                o1_end = old_info(2,1) + o1;
                o2_end = old_info(4,1) + o2;

            end

        else
            DADmethod.Length_of_cell{i,1}(ii,1) = Length_of_cell;   % 150 cells
            DADmethod.Length_of_testcell{i,1}(ii,1) = o2;
        end

        % First 60 cells of each cell are test data
        DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_STEPe2a', i)){ii,1} = ...
            cat(1, DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_STEPe2', i))(o2_start:o2_end,1), ...
            DADmethod.pre_goodepochSet(1:end));

        old_info = [o1_start; o1_end; o2_start; o2_end];
        o1_start = o1_end + 1;
        o1_end = o1_end + o1;
        o2_start = o2_end + 1;
        o2_end = o2_end + o2;
    end

    statsDAD.Length_of_TOTtestcell(i,1) = sum(DADmethod.Length_of_testcell{i,1});
end


%%


% (STEP e3) - Variance within epoch

for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)
        for j = 1:length(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_STEPe2a', i)){k,1})
            DADmethod.epoch_var_STEPe3_FULL{i,1}{k,1}(j,1) = var(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_STEPe2a', i)){k,1}{j,1});
        end
    end
end

for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)
        statsDAD.epoch_var_zscore_STEPe3_FULL{i,1}{k,1} = zscore(DADmethod.epoch_var_STEPe3_FULL{i,1}{k,1}); % Matlab's zscore: z = (data point-pop_mean)/pop_std
    end
end

% --------------------------
% If an epoch is rejected, make the epoch = zero but hold the same data length, 
% by retaining the same data length we can count the correct epochs
% --------------------------
% Checker Program
q = 1;
for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)
        DADmethod.Rej_epoch_STEPe3_FULL{i,1}{k,1} = [];    % Initialize

        for j = 1:length(statsDAD.epoch_var_zscore_STEPe3_FULL{i,1}{k,1})
            if abs(statsDAD.epoch_var_zscore_STEPe3_FULL{i,1}{k,1}(j,1)) > dad.constants.DADepoch_var

                % Reject
                DADmethod.Rej_epoch_STEPe3_FULL{i,1}{k,1}(q,1) = j;     % Counts the rejected epochs location per channel
                q = q + 1;

                %DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe3_FULL', i)){k,1}{j,1} = zeros(2561,1);
                DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe3_FULL', i)){k,1}{j,1} = ...
                    DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_STEPe2a', i)){k,1}{j,1};
            else
                % Accept
                DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe3_FULL', i)){k,1}{j,1} = ...
                    DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_STEPe2a', i)){k,1}{j,1};
            end
        end
        q = 1;
    end
end



% --------------------------
% Keep track of the rejected epochs and at which step they were rejected 0 indicates that the epoch is "good"
% --------------------------
for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)

        for j = 1:length(DADmethod.Rej_epoch_STEPe3_FULL{i,1}{k,1})
            a = DADmethod.Rej_epoch_STEPe3_FULL{i,1}{k,1}(j,1) + testCellsAmt*(k-1);

            if a < (statsDAD.Length_of_TOTtestcell(i,1) + 1)
                if DADmethod.Rej_epoch_STEPe3_FULL{i,1}{k,1}(j,1) < (testCellsAmt+1)
                    statsDAD.rej_struct{i,1}(a,1) = 3;    % Assign a 3 to all rejected epochs to denote (STEP e3)
                end
            end

        end

    end
end


%%


% (STEP e4) - Calculate the Hurst exponent within the epoch

for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)
        for j = 1:length(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe3_FULL', i)){k,1})

            DADmethod.epoch_HE_STEPSe4_FULL{i,1}{k,1}(j,1) = genhurst(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe3_FULL', i)){k,1}{j,1});
            if isnan(DADmethod.epoch_HE_STEPSe4_FULL{i,1}{k,1}(j,1)) == 1
                DADmethod.epoch_HE_STEPSe4_FULL{i,1}{k,1}(j,1) = 0.6;     % An average Hurst value for biological data
            end

        end
    end
end

for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)
        statsDAD.epoch_HE_zscore_STEPe4_FULL{i,1}{k,1} = zscore(DADmethod.epoch_HE_STEPSe4_FULL{i,1}{k,1}); % Matlab's zscore: z = (data point-pop_mean)/pop_std
    end
end


% --------------------------
% If an epoch is rejected, delete the epoch
% An epoch whose amplitude range has a z-score of +/-dad.constants.DADepoch_hurst would be deemed contaminated
% --------------------------
% Checker Program
q = 1;
for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)
        DADmethod.Rej_epoch_STEPe4_FULL{i,1}{k,1} = [];    % Initialize

        for j = 1:length(statsDAD.epoch_HE_zscore_STEPe4_FULL{i,1}{k,1})
            if abs(statsDAD.epoch_HE_zscore_STEPe4_FULL{i,1}{k,1}(j,1)) > dad.constants.DADepoch_hurst

                % Reject
                DADmethod.Rej_epoch_STEPe4_FULL{i,1}{k,1}(q,1) = j;     % Counts the rejected epochs location per channel
                q = q + 1;

                %DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe4_FULL', i)){k,1}{j,1} = zeros(2561,1); % 0, zeros(2561,1)
                DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe4_FULL', i)){k,1}{j,1} = ...
                    DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe3_FULL', i)){k,1}{j,1};
            else
                % Accept
                DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe4_FULL', i)){k,1}{j,1} = ...
                    DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe3_FULL', i)){k,1}{j,1};
            end
        end
        q = 1;
    end
end


% --------------------------
% Keep track of the rejected epochs and at which step they were rejected 0 indicates that the epoch is "good"
% --------------------------
for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)

        for j = 1:length(DADmethod.Rej_epoch_STEPe4_FULL{i,1}{k,1})
            a = DADmethod.Rej_epoch_STEPe4_FULL{i,1}{k,1}(j,1) + testCellsAmt*(k-1);

            if a < (statsDAD.Length_of_TOTtestcell(i,1) + 1)
                if DADmethod.Rej_epoch_STEPe4_FULL{i,1}{k,1}(j,1) < (testCellsAmt+1)
                    statsDAD.rej_struct{i,1}(a,1) = 4;    % Assign a 4 to all rejected epochs to denote (STEP e4)
                end
            end

        end

    end
end


%%


% (STEP e5) - Calculate the Frequency Power of each epoch
stepe5parm.order = statsDAD.order;
stepe5parm.Fs = statsDAD.Fs;

% --------------------------

% 1. (STEP e5a) - Calculate the Frequency Power (delta: 1-3) of each epoch
stepe5parm.filt_bandpass = [1 3];    % low and high frequencies for bandpass filter

for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)
        for j = 1:length(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe4_FULL', i)){k,1})

            if DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe4_FULL', i)){k,1}{j,1} == 0
                DADmethod.FP_STEPe5a_FULL{i,1}{k,1}(j,1) = 0;
            else
                stepe5parm.filtsignal_e5a{i,1}{k,1}{j,1} = PreprocessEEG(stepe5parm, DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe4_FULL', i)){k,1}{j,1});  % filter
                N = length(stepe5parm.filtsignal_e5a{i,1}{k,1}{j,1});
                f = (0:N-1)*statsDAD.Fs/(N-1);   % corresponding frequency vector
                stepe5parm.filtsignal_fft_e5a{i,1}{k,1}{j,1} = fft(stepe5parm.filtsignal_e5a{i,1}{k,1}{j,1}, N);  % Fourier transform
                stepe5parm.filtsignal_fft_gain_e5a{i,1}{k,1}{j,1} = abs(stepe5parm.filtsignal_fft_e5a{i,1}{k,1}{j,1}(1:round(length(f)/2)));  % gain (frequency response)
                DADmethod.FP_STEPe5a_FULL{i,1}{k,1}(j,1) = ((1/(2*pi))*(sum(stepe5parm.filtsignal_fft_gain_e5a{i,1}{k,1}{j,1})))/N;      % signal power
            end

        end
    end
end
% -------------------------- Clear space
stepe5parm = rmfield(stepe5parm, 'filtsignal_e5a');
stepe5parm = rmfield(stepe5parm, 'filtsignal_fft_e5a');
stepe5parm = rmfield(stepe5parm, 'filtsignal_fft_gain_e5a'); 
% --------------------------

% A channel whose Frequency Power had a z-score of +/-2 would be deemed contaminated
for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)
        statsDAD.FP_zscore_STEPe5a_FULL{i,1}{k,1} = zscore(DADmethod.FP_STEPe5a_FULL{i,1}{k,1}); % Matlab's zscore: z = (data point-pop_mean)/pop_std
    end
end

% --------------------------
% If an epoch is rejected, delete the epoch
% --------------------------
% Checker Program
q = 1;
for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)
        DADmethod.Rej_epoch_STEPe5a_FULL{i,1}{k,1} = [];    % Initialize

        for j = 1:length(statsDAD.FP_zscore_STEPe5a_FULL{i,1}{k,1})
            if abs(statsDAD.FP_zscore_STEPe5a_FULL{i,1}{k,1}(j,1)) > dad.constants.DADepoch_deltafreq  % An epoch whose amplitude range has a z-score of +/-2 would be deemed contaminated

                % Reject
                DADmethod.Rej_epoch_STEPe5a_FULL{i,1}{k,1}(q,1) = j;     % Counts the rejected epochs location per channel
                q = q + 1;
                %DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5a_FULL', i)){k,1}{j,1} = zeros(2561,1);
                DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5a_FULL', i)){k,1}{j,1} = ...
                    DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe4_FULL', i)){k,1}{j,1};
            else
                % Accept
                DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5a_FULL', i)){k,1}{j,1} = ...
                    DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe4_FULL', i)){k,1}{j,1};
            end
        end
        q = 1;
    end
end

% --------------------------
% Keep track of the rejected epochs and at which step they were rejected 0 indicates that the epoch is "good"
% --------------------------
for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)

        for j = 1:length(DADmethod.Rej_epoch_STEPe5a_FULL{i,1}{k,1})
            a = DADmethod.Rej_epoch_STEPe5a_FULL{i,1}{k,1}(j,1) + testCellsAmt*(k-1);

            if a < (statsDAD.Length_of_TOTtestcell(i,1) + 1)
                if DADmethod.Rej_epoch_STEPe5a_FULL{i,1}{k,1}(j,1) < (testCellsAmt+1)
                    statsDAD.rej_struct{i,1}(a,1) = 50;    % Assign a 50 to all rejected epochs to denote (STEP e5a)
                end
            end

        end

    end
end

% -------------------------- Clear space
clear i j q a N
for i = 1:dad.NumOfsampledCHs
    DADmethod = rmfield(DADmethod, sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe4_FULL', i));
end
DADmethod = rmfield(DADmethod, 'FP_STEPe5a_FULL');
DADmethod = rmfield(DADmethod, 'Rej_epoch_STEPe5a_FULL');
% --------------------------


%%


% 2. (STEP e5b) - Calculate the Frequency Power (theta: 4-7) of each epoch
stepe5parm.filt_bandpass = [4 7];    % low and high frequencies for bandpass filter

for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)
        for j = 1:length(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5a_FULL', i)){k,1})

            if DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5a_FULL', i)){k,1}{j,1} == 0
                DADmethod.FP_STEPe5b_FULL{i,1}{k,1}(j,1) = 0;
            else
                stepe5parm.filtsignal_e5b{i,1}{k,1}{j,1} = PreprocessEEG(stepe5parm, DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5a_FULL', i)){k,1}{j,1});  % filter
                N = length(stepe5parm.filtsignal_e5b{i,1}{k,1}{j,1});
                f = (0:N-1)*statsDAD.Fs/(N-1);   % corresponding frequency vector
                stepe5parm.filtsignal_fft_e5b{i,1}{k,1}{j,1} = fft(stepe5parm.filtsignal_e5b{i,1}{k,1}{j,1}, N);  % Fourier transform
                stepe5parm.filtsignal_fft_gain_e5b{i,1}{k,1}{j,1} = abs(stepe5parm.filtsignal_fft_e5b{i,1}{k,1}{j,1}(1:round(length(f)/2)));  % gain (frequency response)
                DADmethod.FP_STEPe5b_FULL{i,1}{k,1}(j,1) = ((1/(2*pi))*(sum(stepe5parm.filtsignal_fft_gain_e5b{i,1}{k,1}{j,1})))/N;      % signal power
            end

        end
    end
end
% -------------------------- Clear space
stepe5parm = rmfield(stepe5parm, 'filtsignal_e5b');
stepe5parm = rmfield(stepe5parm, 'filtsignal_fft_e5b');
stepe5parm = rmfield(stepe5parm, 'filtsignal_fft_gain_e5b');
% --------------------------

% A channel whose Frequency Power had a z-score of +/-2 would be deemed contaminated
for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)
        statsDAD.FP_zscore_STEPe5b_FULL{i,1}{k,1} = zscore(DADmethod.FP_STEPe5b_FULL{i,1}{k,1}); % Matlab's zscore: z = (data point-pop_mean)/pop_std
    end
end

% --------------------------
% If an epoch is rejected, delete the epoch
% --------------------------
% Checker Program
q = 1;
for i = 1:dad.NumOfsampledCHs      % 1:18
    for k = 1:DADmethod.times2div(i,1)
        DADmethod.Rej_epoch_STEPe5b_FULL{i,1}{k,1} = [];    % Initialize

        for j = 1:length(statsDAD.FP_zscore_STEPe5b_FULL{i,1}{k,1})
            if abs(statsDAD.FP_zscore_STEPe5b_FULL{i,1}{k,1}(j,1)) > dad.constants.DADepoch_thetafreq  % An epoch whose amplitude range has a z-score of +/-2 would be deemed contaminated

                % Reject
                DADmethod.Rej_epoch_STEPe5b_FULL{i,1}{k,1}(q,1) = j;     % Counts the rejected epochs location per channel
                q = q + 1;
                %DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5b_FULL', i)){k,1}{j,1} = zeros(2561,1);
                DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5b_FULL', i)){k,1}{j,1} = ...
                    DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5a_FULL', i)){k,1}{j,1};
            else
                % Accept
                DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5b_FULL', i)){k,1}{j,1} = ...
                    DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5a_FULL', i)){k,1}{j,1};
            end
        end
        q = 1;
    end
end

% --------------------------
% Keep track of the rejected epochs and at which step they were rejected 0 indicates that the epoch is "good"
% --------------------------
for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)

        for j = 1:length(DADmethod.Rej_epoch_STEPe5b_FULL{i,1}{k,1})
            a = DADmethod.Rej_epoch_STEPe5b_FULL{i,1}{k,1}(j,1) + testCellsAmt*(k-1);

            if a < (statsDAD.Length_of_TOTtestcell(i,1) + 1)
                if DADmethod.Rej_epoch_STEPe5b_FULL{i,1}{k,1}(j,1) < (testCellsAmt+1)
                    statsDAD.rej_struct{i,1}(a,1) = 51;    % Assign a 51 to all rejected epochs to denote (STEP e5b)
                end
            end

        end

    end
end

% -------------------------- Clear space
clear i j k q a N
for i = 1:dad.NumOfsampledCHs
    DADmethod = rmfield(DADmethod, sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5a_FULL', i));
end
DADmethod = rmfield(DADmethod, 'FP_STEPe5b_FULL');
DADmethod = rmfield(DADmethod, 'Rej_epoch_STEPe5b_FULL');
% --------------------------


%%


% 3. (STEP e5c) - Calculate the Frequency Power (alpha: 8-15) of each epoch

stepe5parm.filt_bandpass = [8 15];    % low and high frequencies for bandpass filter

for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)
        for j = 1:length(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5b_FULL', i)){k,1})

            if DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5b_FULL', i)){k,1}{j,1} == 0
                DADmethod.FP_STEPe5c_FULL{i,1}{k,1}(j,1) = 0;
            else
                stepe5parm.filtsignal_e5c{i,1}{k,1}{j,1} = PreprocessEEG(stepe5parm, DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5b_FULL', i)){k,1}{j,1});  % filter
                N = length(stepe5parm.filtsignal_e5c{i,1}{k,1}{j,1});
                f = (0:N-1)*statsDAD.Fs/(N-1);   % corresponding frequency vector
                stepe5parm.filtsignal_fft_e5c{i,1}{k,1}{j,1} = fft(stepe5parm.filtsignal_e5c{i,1}{k,1}{j,1}, N);  % Fourier transform
                stepe5parm.filtsignal_fft_gain_e5c{i,1}{k,1}{j,1} = abs(stepe5parm.filtsignal_e5c{i,1}{k,1}{j,1}(1:round(length(f)/2)));  % gain (frequency response)
                DADmethod.FP_STEPe5c_FULL{i,1}{k,1}(j,1) = ((1/(2*pi))*(sum(stepe5parm.filtsignal_fft_gain_e5c{i,1}{k,1}{j,1})))/N;      % signal power
            end

        end
    end
end
% -------------------------- Clear space
stepe5parm = rmfield(stepe5parm, 'filtsignal_e5c');
stepe5parm = rmfield(stepe5parm, 'filtsignal_fft_e5c');
stepe5parm = rmfield(stepe5parm, 'filtsignal_fft_gain_e5c');
% --------------------------

% A channel whose Frequency Power had a z-score of +/-2.5 would be deemed contaminated
for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)
        statsDAD.FP_zscore_STEPe5c_FULL{i,1}{k,1} = zscore(DADmethod.FP_STEPe5c_FULL{i,1}{k,1});  % Matlab's zscore: z = (data point-pop_mean)/pop_std
    end
end

% --------------------------
% Only tag for sleepiness
% --------------------------
% Checker Program
q = 1;
for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)
        DADmethod.Rej_epoch_STEPe5c_FULL{i,1}{k,1} = [];    % Initialize

        for j = 1:length(statsDAD.FP_zscore_STEPe5c_FULL{i,1}{k,1})
            if abs(statsDAD.FP_zscore_STEPe5c_FULL{i,1}{k,1}(j,1)) > dad.constants.DADepoch_alphafreq  % An epoch whose amplitude range has a z-score of +/-2.5 would be deemed contaminated

                % Reject
                DADmethod.Rej_epoch_STEPe5c_FULL{i,1}{k,1}(q,1) = j;     % Counts the rejected epochs location per channel
                q = q + 1;
                DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5c_FULL', i)){k,1}{j,1} = DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5b_FULL', i)){k,1}{j,1};
            else
                % Accept
                DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5c_FULL', i)){k,1}{j,1} = DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5b_FULL', i)){k,1}{j,1};
            end
        end
        q = 1;
    end
end

% --------------------------
% Keep track of the rejected epochs and at which step they were rejected 0 indicates that the epoch is "good"
% --------------------------
for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)

        for j = 1:length(DADmethod.Rej_epoch_STEPe5c_FULL{i,1}{k,1})
            a = DADmethod.Rej_epoch_STEPe5c_FULL{i,1}{k,1}(j,1) + testCellsAmt*(k-1);

            if a < (statsDAD.Length_of_TOTtestcell(i,1) + 1)
                if DADmethod.Rej_epoch_STEPe5c_FULL{i,1}{k,1}(j,1) < (testCellsAmt+1)
                    statsDAD.rej_struct{i,1}(a,1) = 52;    % Assign a 52 to all rejected epochs to denote (STEP e5c)
                end
            end

        end

    end
end

% -------------------------- Clear space
clear i j k q a N
for i = 1:dad.NumOfsampledCHs
    DADmethod = rmfield(DADmethod, sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5b_FULL', i));
end
DADmethod = rmfield(DADmethod, 'FP_STEPe5c_FULL');
DADmethod = rmfield(DADmethod, 'Rej_epoch_STEPe5c_FULL');
% --------------------------


%%


% 4. (STEP e5d) - Calculate the Frequency Power (beta: 16-31) of each epoch
stepe5parm.filt_bandpass = [16 31];    % low and high frequencies for bandpass filter

for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)
        for j = 1:length(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5c_FULL', i)){k,1})

            if DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5c_FULL', i)){k,1}{j,1} == 0
                DADmethod.FP_STEPe5d_FULL{i,1}{k,1}(j,1) = 0;
            else
                stepe5parm.filtsignal_e5d{i,1}{k,1}{j,1} = PreprocessEEG(stepe5parm, DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5c_FULL', i)){k,1}{j,1});  % filter
                N = length(stepe5parm.filtsignal_e5d{i,1}{k,1}{j,1});
                f = (0:N-1)*statsDAD.Fs/(N-1);   % corresponding frequency vector
                stepe5parm.filtsignal_fft_e5d{i,1}{k,1}{j,1} = fft(stepe5parm.filtsignal_e5d{i,1}{k,1}{j,1}, N);  % Fourier transform
                stepe5parm.filtsignal_fft_gain_e5d{i,1}{k,1}{j,1} = abs(stepe5parm.filtsignal_e5d{i,1}{k,1}{j,1}(1:round(length(f)/2)));  % gain (frequency response)
                DADmethod.FP_STEPe5d_FULL{i,1}{k,1}(j,1) = ((1/(2*pi))*(sum(stepe5parm.filtsignal_fft_gain_e5d{i,1}{k,1}{j,1})))/N;      % signal power
            end

        end

    end
end
% -------------------------- Clear space
stepe5parm = rmfield(stepe5parm, 'filtsignal_e5d');
stepe5parm = rmfield(stepe5parm, 'filtsignal_fft_e5d');
stepe5parm = rmfield(stepe5parm, 'filtsignal_fft_gain_e5d');
% -------------------------- 

% A channel whose Frequency Power had a z-score of +/-2.5 would be deemed contaminated
for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)
        statsDAD.FP_zscore_STEPe5d_FULL{i,1}{k,1} = zscore(DADmethod.FP_STEPe5d_FULL{i,1}{k,1});  % Matlab's zscore: z = (data point-pop_mean)/pop_std
    end
end

% --------------------------
% Only tag for mental activity
% --------------------------
% Checker Program
q = 1;
for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)
        DADmethod.Rej_epoch_STEPe5d_FULL{i,1}{k,1} = [];    % Initialize

        for j = 1:length(statsDAD.FP_zscore_STEPe5d_FULL{i,1}{k,1})
            if abs(statsDAD.FP_zscore_STEPe5d_FULL{i,1}{k,1}(j,1)) > dad.constants.DADepoch_betafreq  % An epoch whose amplitude range has a z-score of +/-2.5 would be deemed contaminated

                % Reject
                DADmethod.Rej_epoch_STEPe5d_FULL{i,1}{k,1}(q,1) = j;     % Counts the rejected epochs location per channel
                q = q + 1;
                DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5d_FULL', i)){k,1}{j,1} = DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5c_FULL', i)){k,1}{j,1};
            else
                % Accept
                DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5d_FULL', i)){k,1}{j,1} = DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5c_FULL', i)){k,1}{j,1};
            end
        end
        q = 1;
    end
end

% --------------------------
% Keep track of the rejected epochs and at which step they were rejected 0 indicates that the epoch is "good"
% --------------------------
for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)

        for j = 1:length(DADmethod.Rej_epoch_STEPe5d_FULL{i,1}{k,1})
            a = DADmethod.Rej_epoch_STEPe5d_FULL{i,1}{k,1}(j,1) + testCellsAmt*(k-1);

            if a < (statsDAD.Length_of_TOTtestcell(i,1) + 1)
                if DADmethod.Rej_epoch_STEPe5d_FULL{i,1}{k,1}(j,1) < (testCellsAmt+1)
                    statsDAD.rej_struct{i,1}(a,1) = 53;    % Assign a 53 to all rejected epochs to denote (STEP e5d)
                end
            end

        end

    end
end

% -------------------------- Clear space
clear i j k q a N
for i = 1:dad.NumOfsampledCHs
    DADmethod = rmfield(DADmethod, sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5c_FULL', i));
end
DADmethod = rmfield(DADmethod, 'FP_STEPe5d_FULL');
DADmethod = rmfield(DADmethod, 'Rej_epoch_STEPe5d_FULL');
% --------------------------


%%


% 5. (STEP e5e) - Calculate the Frequency Power (gamma: 32-49) of each epoch
stepe5parm.filt_bandpass = [32 49];    % low and high frequencies for bandpass filter

for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)
        for j = 1:length(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5d_FULL', i)){k,1})

            if DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5d_FULL', i)){k,1}{j,1} == 0
                DADmethod.FP_STEPe5e_FULL{i,1}{k,1}(j,1) = 0;
            else
                stepe5parm.filtsignal_e5e{i,1}{k,1}{j,1} = PreprocessEEG(stepe5parm, DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5d_FULL', i)){k,1}{j,1});  % filter
                N = length(stepe5parm.filtsignal_e5e{i,1}{k,1}{j,1});
                f = (0:N-1)*statsDAD.Fs/(N-1);   % corresponding frequency vector
                stepe5parm.filtsignal_fft_e5e{i,1}{k,1}{j,1} = fft(stepe5parm.filtsignal_e5e{i,1}{k,1}{j,1}, N);  % Fourier transform
                stepe5parm.filtsignal_fft_gain_e5e{i,1}{k,1}{j,1} = abs(stepe5parm.filtsignal_fft_e5e{i,1}{k,1}{j,1}(1:round(length(f)/2)));  % gain (frequency response)
                DADmethod.FP_STEPe5e_FULL{i,1}{k,1}(j,1) = ((1/(2*pi))*(sum(stepe5parm.filtsignal_fft_gain_e5e{i,1}{k,1}{j,1})))/N;      % signal power
            end
        end

    end
end
% -------------------------- Clear space
clear stepe5parm
% -------------------------- 

% A channel whose Frequency Power had a z-score of +/-2 would be deemed contaminated
for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)
        statsDAD.FP_zscore_STEPe5e_FULL{i,1}{k,1} = zscore(DADmethod.FP_STEPe5e_FULL{i,1}{k,1}); % Matlab's zscore: z = (data point-pop_mean)/pop_std
    end
end

% --------------------------
% If an epoch is rejected, delete the epoch
% --------------------------
% Checker Program
q = 1;
for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)
        DADmethod.Rej_epoch_STEPe5e_FULL{i,1}{k,1} = [];    % Initialize

        for j = 1:length(statsDAD.FP_zscore_STEPe5e_FULL{i,1}{k,1})
            if abs(statsDAD.FP_zscore_STEPe5e_FULL{i,1}{k,1}(j,1)) > dad.constants.DADepoch_gammafreq  % An epoch whose amplitude range has a z-score of +/-2 would be deemed contaminated

                % Reject
                DADmethod.Rej_epoch_STEPe5e_FULL{i,1}{k,1}(q,1) = j;     % Counts the rejected epochs location per channel
                q = q + 1;
                %DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5e_FULL', i)){k,1}{j,1} = zeros(2561,1);
                DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5e_FULL', i)){k,1}{j,1} = ...
                    DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5d_FULL', i)){k,1}{j,1};
            else
                % Accept
                DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5e_FULL', i)){k,1}{j,1} = ...
                    DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5d_FULL', i)){k,1}{j,1};
            end
        end
        q = 1;
    end
end

% --------------------------
% Keep track of the rejected epochs and at which step they were rejected 0 indicates that the epoch is "good"
% --------------------------
for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)

        for j = 1:length(DADmethod.Rej_epoch_STEPe5e_FULL{i,1}{k,1})
            a = DADmethod.Rej_epoch_STEPe5e_FULL{i,1}{k,1}(j,1) + testCellsAmt*(k-1);

            if a < (statsDAD.Length_of_TOTtestcell(i,1) + 1)
                if DADmethod.Rej_epoch_STEPe5e_FULL{i,1}{k,1}(j,1) < (testCellsAmt+1)
                    statsDAD.rej_struct_gamma{i,1}(a,1) = 54;
                    statsDAD.rej_struct{i,1}(a,1) = 54;
                end
            end

        end

    end
end

% -------------------------- Clear space
clear i j k q a N
for i = 1:dad.NumOfsampledCHs
    DADmethod = rmfield(DADmethod, sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5d_FULL', i));
end
DADmethod = rmfield(DADmethod, 'FP_STEPe5e_FULL');
DADmethod = rmfield(DADmethod, 'Rej_epoch_STEPe5e_FULL');
% --------------------------


%%


% (STEP e6) - Peak within epoch

for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)
        for j = 1:length(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5e_FULL', i)){k,1})

            sel = ( max(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5e_FULL', i)){k,1}{j,1}) - ...
                min(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5e_FULL', i)){k,1}{j,1}) )/4;

            thresh = max(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5e_FULL', i)){k,1}{j,1})*0.55;

            [DADmethod.epoch_peakLoc_STEPe6_FULL{i,1}{k,1}{j,1}, DADmethod.epoch_peakMag_STEPe6_FULL{i,1}{k,1}{j,1}] = ...
                peakfinder(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5e_FULL', i)){k,1}{j,1}, sel, thresh);

            % Number of peaks present
            DADmethod.epoch_peaks_STEPe6_FULL{i,1}{k,1}(j,1) = length(DADmethod.epoch_peakLoc_STEPe6_FULL{i,1}{k,1}{j,1});

        end
    end
end




for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)
        statsDAD.epoch_peaks_zscore_STEPe6_FULL{i,1}{k,1} = zscore(DADmethod.epoch_peaks_STEPe6_FULL{i,1}{k,1}); 
        % Matlab's zscore: z = (data point-pop_mean)/pop_std
    end
end



% --------------------------
% If an epoch is rejected, make the epoch = zero but hold the same data length, 
% by retaining the same data length we can count the correct epochs
% --------------------------
% Checker Program
q = 1;
for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)
        DADmethod.Rej_epoch_STEPe6_FULL{i,1}{k,1} = [];    % Initialize

        for j = 1:length(statsDAD.epoch_peaks_zscore_STEPe6_FULL{i,1}{k,1})
            if abs(statsDAD.epoch_peaks_zscore_STEPe6_FULL{i,1}{k,1}(j,1)) > dad.constants.DADepoch_peaks

                % Reject
                DADmethod.Rej_epoch_STEPe6_FULL{i,1}{k,1}(q,1) = j;     % Counts the rejected epochs location per channel
                q = q + 1;

                %DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe6_FULL', i)){k,1}{j,1} = zeros(2561,1);
                DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe6_FULL', i)){k,1}{j,1} = ...
                    DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5e_FULL', i)){k,1}{j,1};
            else
                % Accept
                DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe6_FULL', i)){k,1}{j,1} = ...
                    DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5e_FULL', i)){k,1}{j,1};
            end
        end
        q = 1;
    end
end


% --------------------------
% Keep track of the rejected epochs and at which step they were rejected 0 indicates that the epoch is "good"
% --------------------------
for i = 1:dad.NumOfsampledCHs
    for k = 1:DADmethod.times2div(i,1)

        for j = 1:length(DADmethod.Rej_epoch_STEPe6_FULL{i,1}{k,1})
            a = DADmethod.Rej_epoch_STEPe6_FULL{i,1}{k,1}(j,1) + testCellsAmt*(k-1);

            if a < (statsDAD.Length_of_TOTtestcell(i,1) + 1)
                
                if DADmethod.Rej_epoch_STEPe6_FULL{i,1}{k,1}(j,1) < (testCellsAmt+1)
                    
                    if statsDAD.rej_struct_gamma{i,1}(a,1) == 54
                        statsDAD.rej_struct{i,1}(a,1) = 6;    % Assign a 6 to all rejected epochs to denote (STEP e6)
                    end
                    
                end
                
            end

        end

    end
end

% -------------------------- Clear space
clear i k j q a counter
for i = 1:dad.NumOfsampledCHs
    DADmethod = rmfield(DADmethod, sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5e_FULL', i));
end
DADmethod = rmfield(DADmethod, 'epoch_peaks_STEPe6_FULLall');
DADmethod = rmfield(DADmethod, 'epoch_peaks_STEPe6_FULL');
DADmethod = rmfield(DADmethod, 'Rej_epoch_STEPe6_FULL');
% --------------------------

%%
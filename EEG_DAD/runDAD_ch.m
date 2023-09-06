function runDAD_ch

% If dad.mat exists, load it
if exist('dad.mat', 'file') == 2
    load dad.mat
end

global dad

dad.popup_splocValue = get(dad.popup_sploc, 'Value');
temp.data_eegch = dad.data_eegchall{dad.popup_splocValue,1}; % Everytime the menu is changed, update the data vector


% Check if database is loaded
if dad.database == 0  % No database has been loaded
    errordlg('Load a Database (Import tab)...')
    
else
    
    % ****************************
    h = waitbar(0, 'Computing DAD (whole)...');
    steps = 31;  % Total number of steps to complete the computation
    % ****************************
    
    % ----------------------------------
    % Calculate the DAD method 
    % ----------------------------------

    % (STEP 00) - Select the database 
    % Transform matrix of past data into cell array
    [a1, a2] = size(dad.database);      %#ok<ASGLU> %a2 is number of past data sets
    for EXPs = 1:a2
        DADmethod.database{EXPs, 1} = dad.database(:, EXPs);
    end
    
    DADmethod.NumOfEXPs = a2+1; % Total number of datasets to compare at one time
    DADmethod.NumOfCHs = 1;     % For now just loading 1 spatial channel at a time
    
    % Add the new dataset at the end
    DADmethod.database{DADmethod.NumOfEXPs, 1} = temp.data_eegch;
    
    % ****************************
    curstep = 1;
    waitbar(curstep/steps);
    % ****************************



    % (STEP 0) - Filter: Bandpass filter the data to remove frequencies greater than 50 Hz and lower than 1Hz
    fid = fopen('history.txt', 'at');
    fprintf(fid, 'DAD (STEP 0) - Filter: Bandpass filter the data to remove frequencies greater than 50 Hz and lower than 1Hz\n');
    fclose(fid);

    statsDAD.order = 4;
    statsDAD.filt_bandpass = [1 50];    % low and high frequencies for bandpass filter
    statsDAD.Fs = dad.data.samplingrate;	 % 256


    for i = 1:DADmethod.NumOfEXPs
        DADmethod.filtsignal{i,1} = PreprocessEEG(statsDAD, DADmethod.database{i,1});
    end

    DADmethod.vec_Acc_STEPlast = DADmethod.filtsignal;

    dad.NumOfsampledCHs = DADmethod.NumOfEXPs*DADmethod.NumOfCHs;    % 16 => 2 channels sampled 8 times each

    % -------------------------- Clear space
    DADmethod = rmfield(DADmethod, 'filtsignal');
    % -------------------------- 
    
    % ****************************
    curstep = 2;
    waitbar(curstep/steps);
    % ****************************


    fid = fopen('history.txt', 'at');
    fprintf(fid, 'The entire dataset will be evaluated before epochs...\n');
    fclose(fid);


    % ooooooooooooooooooooooooooooo
    % Section: Channels artifacts
    % ooooooooooooooooooooooooooooo

    % --------------------------
    % (STEP c1) - Calculate the variance of each channel
    fid = fopen('history.txt', 'at');
    fprintf(fid, '(STEP c1) - Calculate the variance of each channel\n');
    fclose(fid);

    for i = 1:dad.NumOfsampledCHs
        DADmethod.varCH_STEPc1(i,1) = var(DADmethod.vec_Acc_STEPlast{i,1});
    end

    % A channel whose variance had a z-score of +/-3 would be deemed contaminated
    statsDAD.varCH_zscore_STEPc1 = zscore(DADmethod.varCH_STEPc1); % Matlab's zscore: z = (data point-pop_mean)/pop_std

    fid = fopen('history.txt', 'at');
    for i = 1:length(statsDAD.varCH_zscore_STEPc1)
        fprintf(fid, '\t%.2f\t\n', statsDAD.varCH_zscore_STEPc1(i,1));
    end
    fclose(fid);


    % 1 channel needs at least 7 experimental data sets to compare with to get a good estimate of the
    % channel reliability in the context of the entire experiment 

    % --------------------------
    % If a channel is rejected, dialate and interpolate the channel. Interpolate only to compare with dialation.
    % --------------------------

    % Checker Program
    Acc = 1;
    Rej = 1;
    for i = 1:length(statsDAD.varCH_zscore_STEPc1)
        if abs(statsDAD.varCH_zscore_STEPc1(i,1)) < 3
            % Accept
            DADmethod.vec_Acc_STEPc1{Acc,1} = DADmethod.vec_Acc_STEPlast{i,1};
            Acc = Acc + 1;
            
            statsDAD.rej_struct_whole{i,1}(1,1) = 0;
        else
            % Reject
            DADmethod.vec_Rej_STEPc1{Rej,1} = DADmethod.vec_Acc_STEPlast{i,1};
            Rej_chan_STEPc1(Rej,1) = i; %#ok<AGROW>
            Rej = Rej + 1;
            
            statsDAD.rej_struct_whole{i,1}(1,1) = 3;
        end
    end

    if exist('Rej_chan_STEPc1', 'var')  ~= 0    % if it exists
        fid = fopen('history.txt', 'at');
        fprintf(fid, 'Rejected channels: %d\n', Rej_chan_STEPc1);
        fclose(fid);
        DADmethod.Rej_chan_STEPc1 = Rej_chan_STEPc1;
    end
    
    
    
    % Reset the length of the number of sampled channels, in case channels were rejected
    dad.NumOfsampledCHs = length(DADmethod.vec_Acc_STEPc1);

    % -------------------------- Clear space
    clear i Acc Rej Rej_chan_STEPc1
    DADmethod = rmfield(DADmethod, 'vec_Acc_STEPlast');
    % -------------------------- 
    
    % ****************************
    curstep = 3;
    waitbar(curstep/steps);
    % ****************************




    % --------------------------
    fid = fopen('history.txt', 'at');
    fprintf(fid, '(STEP c2) - Calculate the median of each channel\n');
    fclose(fid);
    % (STEP 3) - Calculate the median of each channel: slope/gradient for channel - realistic eeg data has drift, but bad signals data has a lot of data around the median (no drift)

    for i = 1:dad.NumOfsampledCHs
        DADmethod.medianCH_STEPc2(i,1) = median(DADmethod.vec_Acc_STEPc1{i,1});
    end

    % A channel whose variance had a z-score of +/-3 would be deemed contaminated
    statsDAD.medianCH_zscore_STEPc2 = zscore(DADmethod.medianCH_STEPc2); % Matlab's zscore: z = (data point-pop_mean)/pop_std

    fid = fopen('history.txt', 'at');
    for i = 1:length(statsDAD.medianCH_zscore_STEPc2)
        fprintf(fid, '\t%.2f\t\n', statsDAD.medianCH_zscore_STEPc2(i,1));
    end
    fclose(fid);


    % --------------------------
    % If a channel is rejected, delete channel - we delete because the signal is not a realistic biological signal (it is noise).
    % --------------------------

    % Checker Program
    Acc = 1;
    Rej = 1;
    for i = 1:length(statsDAD.medianCH_zscore_STEPc2)

        if abs(statsDAD.medianCH_zscore_STEPc2(i,1)) < 2.39
            % Accept
            DADmethod.vec_Acc_STEPc2{Acc,1} = DADmethod.vec_Acc_STEPc1{i,1};
            Acc = Acc + 1;
            
            statsDAD.rej_struct_whole{i,1}(2,1) = 0;
        else
            % Reject
            DADmethod.vec_Rej_STEPc2{Rej,1} = DADmethod.vec_Acc_STEPc1{i,1};

            Rej_chan_STEPc2(Rej,1) = i; %#ok<AGROW>
            Rej = Rej + 1;
            
            statsDAD.rej_struct_whole{i,1}(2,1) = 1;
        end

    end

    if exist('Rej_chan_STEPc2', 'var') ~= 0
        fid = fopen('history.txt', 'at');
        fprintf(fid, 'Rejected channels: %d\n', Rej_chan_STEPc2);
        fclose(fid);
        DADmethod.Rej_chan_STEPc2 = Rej_chan_STEPc2;
    end

    % Reset the length of the number of sampled channels, in case channels were rejected
    dad.NumOfsampledCHs = length(DADmethod.vec_Acc_STEPc2);

    % -------------------------- Clear space
    clear i Acc Rej Rej_chan_STEPc2
    DADmethod = rmfield(DADmethod, 'vec_Acc_STEPc1');
    DADmethod = rmfield(DADmethod, 'varCH_STEPc1');
    % -------------------------- 
    
    % ****************************
    curstep = 4;
    waitbar(curstep/steps);
    % ****************************




    % % --------------------------
    % fid = fopen('history.txt', 'at');
    % fprintf(fid, '(STEP c3) - Calculate the correlation between past data from the same channel via Pearson correlation\n');
    % fclose(fid);
    % % (STEP c3) - Calculate the correlation between past data from the same channel via Pearson correlation. 
    % % And the correlation between the other channels and their past data via Pearson correlation.
    % 
    % for i = 1:dad.NumOfsampledCHs
    %     for j = 1:dad.NumOfsampledCHs
    %         minval = min([length(DADmethod.vec_Acc_STEPc2{i,1}); length(DADmethod.vec_Acc_STEPc2{j,1})]);
    %         
    %         DADmethod.rho_STEPc3(i,j) = corr(DADmethod.vec_Acc_STEPc2{i,1}(1:minval)', DADmethod.vec_Acc_STEPc2{j,1}(1:minval)', 'type', 'Pearson');
    %     end
    % end
    % 
    % % -------------------------- Clear space
    % clear res minval i j
    % DADmethod = rmfield(DADmethod, 'medianCH_STEPc2');
    % % -------------------------- 





    % % --------------------------
    % fid = fopen('history.txt', 'at');
    % fprintf(fid, '(STEP c4) - Calculate the mean correlation coefficient of channels\n');
    % fclose(fid);
    % % (STEP c4) - Calculate the mean correlation coefficient of channels
    % 
    % DADmethod.meanCorr_STEPc4 = sum(DADmethod.rho_STEPc3, 2);
    % 
    % % A channel whose mean correlation has a z-score of +/-3 would be deemed contaminated
    % statsDAD.meanCorr_zscore_STEPc4 = zscore(DADmethod.meanCorr_STEPc4); % Matlab's zscore: z = (data point-pop_mean)/pop_std
    % 
    % fid = fopen('history.txt', 'at');
    % for i = 1:length(statsDAD.meanCorr_zscore_STEPc4)
    %     fprintf(fid, '\t%.2f\t\n', statsDAD.meanCorr_zscore_STEPc4(i,1));
    % end
    % fclose(fid);
    % 
    % 
    % % --------------------------
    % % If a channel is rejected, dialate and interpolate the channel. Interpolate only to compare with dialation.
    % % --------------------------
    % 
    % % Checker Program
    % Rej = 1;
    % for i = 1:length(statsDAD.meanCorr_zscore_STEPc4)
    %     
    %     if abs(statsDAD.meanCorr_zscore_STEPc4(i,1)) < 3
    %         % Accept
    %         DADmethod.vec_Acc_STEPc4{i,1} = DADmethod.vec_Acc_STEPc2{i,1};
    %
    %         statsDAD.rej_struct_whole{i,1}(3,1) = 0;
    %     else
    %         % Reject
    %         DADmethod.vec_Rej_STEPc4{Rej,1} = DADmethod.vec_Acc_STEPc2{i,1};
    %         
    %         % Alter the Rejected signal using dialation
    %         DADmethod.vec_Acc_STEPc4{i,1} = JAR_dialate(DADmethod.vec_Rej_STEPc4{Rej,1});
    %         
    %         Rej_chan_STEPc4(Rej,1) = i; %#ok<AGROW>
    %         
    %         Rej = Rej + 1;
    %         statsDAD.rej_struct_whole{i,1}(3,1) = 2;
    %     end
    % end
    % 
    % if exist('Rej_chan_STEPc4', 'var')  ~= 0
    %     fid = fopen('history.txt', 'at');
    %     fprintf(fid, 'Rejected channels: %d\n', DADmethod.Rej_chan_STEPc4);
    %     fclose(fid);
    %     DADmethod.Rej_chan_STEPc4 = Rej_chan_STEPc4;
    % end
    % 
    % % -------------------------- Clear space
    % clear i Rej Rej_chan_STEPc4
    % DADmethod = rmfield(DADmethod, 'vec_Acc_STEPc2');
    % DADmethod = rmfield(DADmethod, 'rho_STEPc3');
    % DADmethod = rmfield(DADmethod, 'meanCorr_STEPc4');
    % % -------------------------- 










    fid = fopen('history.txt', 'at');
    fprintf(fid, '(STEP c5) - Calculate the Hurst exponent of each channel\n');
    fclose(fid);
    % (STEP c5) - Calculate the Hurst exponent of each channel
    for i = 1:dad.NumOfsampledCHs
        DADmethod.HE_STEPc5(i,1) = genhurst(DADmethod.vec_Acc_STEPc2{i,1});
    end

    % A channel whose Hurst Exponent had a z-score of +/-3 would be deemed contaminated
    statsDAD.HE_zscore_STEPc5 = zscore(DADmethod.HE_STEPc5); % Matlab's zscore: z = (data point-pop_mean)/pop_std

    fid = fopen('history.txt', 'at');
    for i = 1:length(statsDAD.HE_zscore_STEPc5)
        fprintf(fid, '\t%.2f\t\n', statsDAD.HE_zscore_STEPc5(i,1));
    end
    fclose(fid);


    % --------------------------
    % If a channel is rejected, delete channel - we delete because the signal is not a realistic biological signal (it is noise).
    % --------------------------

    % Checker Program
    Acc = 1;
    Rej = 1;
    for i = 1:length(statsDAD.HE_zscore_STEPc5)

        if abs(statsDAD.HE_zscore_STEPc5(i,1)) < 2.2
            % Accept
            DADmethod.vec_Acc_STEPc5{Acc,1} = DADmethod.vec_Acc_STEPc2{i,1};
            Acc = Acc + 1;
            
            statsDAD.rej_struct_whole{i,1}(4,1) = 0;
        else
            % Reject
            DADmethod.vec_Rej_STEPc5{Rej,1} = DADmethod.vec_Acc_STEPc2{i,1};
            Rej_chan_STEPc5(Rej,1) = i; %#ok<AGROW>
            Rej = Rej + 1;
            
            statsDAD.rej_struct_whole{i,1}(4,1) = 4;
        end

    end
    
    % ****************************
    curstep = 5;
    waitbar(curstep/steps);
    % ****************************
    
    
    % -------------------------- Clear space
    DADmethod = rmfield(DADmethod, 'vec_Acc_STEPc2');
    % -------------------------- 


    if exist('Rej_chan_STEPc5', 'var') ~= 0
        fid = fopen('history.txt', 'at');
        fprintf(fid, 'Rejected channels: %d\n', Rej_chan_STEPc5);
        fclose(fid);
        DADmethod.Rej_chan_STEPc5 = Rej_chan_STEPc5;
    end

    % Reset the length of the number of sampled channels, in case channels were rejected
    dad.NumOfsampledCHs = length(DADmethod.vec_Acc_STEPc5);

    % -------------------------- Clear space
    clear i Acc Rej Rej_chan_STEPc5
    % DADmethod = rmfield(DADmethod, 'vec_Acc_STEPc4');
    DADmethod = rmfield(DADmethod, 'HE_STEPc5');
    % -------------------------- 



    % Pass the last datasets to a general variable
    DADmethod.vec_Acc_STEPlast =  DADmethod.vec_Acc_STEPc5;
    
    % ****************************
    curstep = 6;
    waitbar(curstep/steps);
    % ****************************




    fid = fopen('history.txt', 'at');
    fprintf(fid, '(STEP c6) - Calculate the Frequency Power of each channel\n');
    fclose(fid);

    stepc6parm.order = statsDAD.order;
    stepc6parm.Fs = statsDAD.Fs;

    % --------------------------

    % 1. (STEP c6a) - Calculate the Frequency Power (delta: 0-3) of each channel
    fid = fopen('history.txt', 'at');
    fprintf(fid, '(STEP c6a) - Calculate the Frequency Power (delta: 0-3) of each channel\n');
    fclose(fid);

    stepc6parm.filt_bandpass = [1 3];    % low and high frequencies for bandpass filter

    for i = 1:dad.NumOfsampledCHs
        stepc6parm.filtsignal_c6a{i,1} = PreprocessEEG(stepc6parm, DADmethod.vec_Acc_STEPc5{i,1});  % filter
        N = length(stepc6parm.filtsignal_c6a{i,1});
        f = (0:N-1)*statsDAD.Fs/(N-1);   % corresponding frequency vector
        stepc6parm.filtsignal_fft_c6a{i,1} = fft(stepc6parm.filtsignal_c6a{i,1}, N);  % Fourier transform
        stepc6parm.filtsignal_fft_gain_c6a{i,1} = abs(stepc6parm.filtsignal_fft_c6a{i,1}(1:round(length(f)/2)));  % gain (frequency response)
        DADmethod.FP_STEPc6a(i,1) = ((1/(2*pi))*(sum(stepc6parm.filtsignal_fft_gain_c6a{i,1})))/N;      % signal power
    end
    % -------------------------- Clear space
    stepc6parm = rmfield(stepc6parm, 'filtsignal_c6a');
    stepc6parm = rmfield(stepc6parm, 'filtsignal_fft_c6a');
    stepc6parm = rmfield(stepc6parm, 'filtsignal_fft_gain_c6a');
    % -------------------------- 

    % A channel whose Frequency Power had a z-score of +/-3 would be deemed contaminated
    statsDAD.FP_zscore_STEPc6a = zscore(DADmethod.FP_STEPc6a); % Matlab's zscore: z = (data point-pop_mean)/pop_std

    fid = fopen('history.txt', 'at');
    for i = 1:length(statsDAD.FP_zscore_STEPc6a)
        fprintf(fid, '\t%.2f\t\n', statsDAD.FP_zscore_STEPc6a(i,1));
    end
    fclose(fid);
    
    
    % --------------------------
    % If a channel is rejected, delete channel - we delete because the signal is not a realistic biological signal (it is noise).
    % --------------------------
    % Checker Program
    Acc = 1;
    Rej = 1;
    for i = 1:length(statsDAD.FP_zscore_STEPc6a)

        if abs(statsDAD.FP_zscore_STEPc6a(i,1)) < 2.5
            % Accept
            DADmethod.vec_Acc_STEPc6a{Acc,1} = DADmethod.vec_Acc_STEPc5{i,1};
            Acc = Acc + 1;
            
            statsDAD.rej_struct_whole{i,1}(5,1) = 0;
        else
            % Reject
            DADmethod.vec_Rej_STEPc6a{Rej,1} = DADmethod.vec_Acc_STEPc5{i,1};
            Rej_chan_STEPc6a(Rej,1) = i; %#ok<AGROW>
            Rej = Rej + 1;
            
            statsDAD.rej_struct_whole{i,1}(5,1) = 50;
        end

    end
    
    if exist('Rej_chan_STEPc6a', 'var') ~= 0
        fid = fopen('history.txt', 'at');
        fprintf(fid, 'Rejected channels: %d\n', Rej_chan_STEPc6a);
        fclose(fid);
        DADmethod.Rej_chan_STEPc6a = Rej_chan_STEPc6a;
    end

    % Reset the length of the number of sampled channels, in case channels were rejected
    dad.NumOfsampledCHs = length(DADmethod.vec_Acc_STEPc6a);
    
    
    % ****************************
    curstep = 7;
    waitbar(curstep/steps);
    % ****************************






    % 2. (STEP c6b) - Calculate the Frequency Power (theta: 4-7) of each channel
    fid = fopen('history.txt', 'at');
    fprintf(fid, '(STEP c6b) - Calculate the Frequency Power (theta: 4-7) of each channel\n');
    fclose(fid);

    stepc6parm.filt_bandpass = [4 7];    % low and high frequencies for bandpass filter

    for i = 1:dad.NumOfsampledCHs
        stepc6parm.filtsignal_c6b{i,1} = PreprocessEEG(stepc6parm, DADmethod.vec_Acc_STEPc5{i,1});  % filter
        N = length(stepc6parm.filtsignal_c6b{i,1});
        f = (0:N-1)*statsDAD.Fs/(N-1);   % corresponding frequency vector
        stepc6parm.filtsignal_fft_c6b{i,1} = fft(stepc6parm.filtsignal_c6b{i,1}, N);  % Fourier transform
        stepc6parm.filtsignal_fft_gain_c6b{i,1} = abs(stepc6parm.filtsignal_fft_c6b{i,1}(1:round(length(f)/2)));  % gain (frequency response)
        DADmethod.FP_STEPc6b(i,1) = ((1/(2*pi))*(sum(stepc6parm.filtsignal_fft_gain_c6b{i,1})))/N;      % signal power
    end
    % -------------------------- Clear space
    stepc6parm = rmfield(stepc6parm, 'filtsignal_c6b');
    stepc6parm = rmfield(stepc6parm, 'filtsignal_fft_c6b');
    stepc6parm = rmfield(stepc6parm, 'filtsignal_fft_gain_c6b');
    % -------------------------- 

    % A channel whose Frequency Power had a z-score of +/-3 would be deemed contaminated
    statsDAD.FP_zscore_STEPc6b = zscore(DADmethod.FP_STEPc6b); % Matlab's zscore: z = (data point-pop_mean)/pop_std

    fid = fopen('history.txt', 'at');
    for i = 1:length(statsDAD.FP_zscore_STEPc6b)
        fprintf(fid, '\t%.2f\t\n', statsDAD.FP_zscore_STEPc6b(i,1));
    end
    fclose(fid);
    
    
    
    % ****************************
    curstep = 8;
    waitbar(curstep/steps);
    % ****************************







    % 3. (STEP c6c) - Calculate the Frequency Power (alpha: 8-15) of each channel
    fid = fopen('history.txt', 'at');
    fprintf(fid, '(STEP c6c) - Calculate the Frequency Power (alpha: 8-15) of each channel\n');
    fclose(fid);

    stepc6parm.filt_bandpass = [8 15];    % low and high frequencies for bandpass filter

    for i = 1:dad.NumOfsampledCHs
        stepc6parm.filtsignal_c6c{i,1} = PreprocessEEG(stepc6parm, DADmethod.vec_Acc_STEPc5{i,1});  % filter
        N = length(stepc6parm.filtsignal_c6c{i,1});
        f = (0:N-1)*statsDAD.Fs/(N-1);   % corresponding frequency vector
        stepc6parm.filtsignal_fft_c6c{i,1} = fft(stepc6parm.filtsignal_c6c{i,1}, N);  % Fourier transform
        stepc6parm.filtsignal_fft_gain_c6c{i,1} = abs(stepc6parm.filtsignal_fft_c6c{i,1}(1:round(length(f)/2)));  % gain (frequency response)
        DADmethod.FP_STEPc6c(i,1) = ((1/(2*pi))*(sum(stepc6parm.filtsignal_fft_gain_c6c{i,1})))/N;      % signal power
    end
    % -------------------------- Clear space
    stepc6parm = rmfield(stepc6parm, 'filtsignal_c6c');
    stepc6parm = rmfield(stepc6parm, 'filtsignal_fft_c6c');
    stepc6parm = rmfield(stepc6parm, 'filtsignal_fft_gain_c6c');
    % -------------------------- 

    % A channel whose Frequency Power had a z-score of +/-3 would be deemed contaminated
    statsDAD.FP_zscore_STEPc6c = zscore(DADmethod.FP_STEPc6c); % Matlab's zscore: z = (data point-pop_mean)/pop_std

    fid = fopen('history.txt', 'at');
    for i = 1:length(statsDAD.FP_zscore_STEPc6c)
        fprintf(fid, '\t%.2f\t\n', statsDAD.FP_zscore_STEPc6c(i,1));
    end
    fclose(fid);
    
    % ****************************
    curstep = 9;
    waitbar(curstep/steps);
    % ****************************







    % 4. (STEP c6d) - Calculate the Frequency Power (beta: 16-31) of each channel
    fid = fopen('history.txt', 'at');
    fprintf(fid, '(STEP c6d) - Calculate the Frequency Power (beta: 16-31) of each channel\n');
    fclose(fid);

    stepc6parm.filt_bandpass = [16 31];    % low and high frequencies for bandpass filter

    for i = 1:dad.NumOfsampledCHs
        stepc6parm.filtsignal_c6d{i,1} = PreprocessEEG(stepc6parm, DADmethod.vec_Acc_STEPc5{i,1});  % filter
        N = length(stepc6parm.filtsignal_c6d{i,1});
        f = (0:N-1)*statsDAD.Fs/(N-1);   % corresponding frequency vector
        stepc6parm.filtsignal_fft_c6d{i,1} = fft(stepc6parm.filtsignal_c6d{i,1}, N);  % Fourier transform
        stepc6parm.filtsignal_fft_gain_c6d{i,1} = abs(stepc6parm.filtsignal_fft_c6d{i,1}(1:round(length(f)/2)));  % gain (frequency response)
        DADmethod.FP_STEPc6d(i,1) = ((1/(2*pi))*(sum(stepc6parm.filtsignal_fft_gain_c6d{i,1})))/N;      % signal power
    end
    % -------------------------- Clear space
    stepc6parm = rmfield(stepc6parm, 'filtsignal_c6d');
    stepc6parm = rmfield(stepc6parm, 'filtsignal_fft_c6d');
    stepc6parm = rmfield(stepc6parm, 'filtsignal_fft_gain_c6d');
    % --------------------------

    % A channel whose Frequency Power had a z-score of +/-3 would be deemed contaminated
    statsDAD.FP_zscore_STEPc6d = zscore(DADmethod.FP_STEPc6d); % Matlab's zscore: z = (data point-pop_mean)/pop_std

    fid = fopen('history.txt', 'at');
    for i = 1:length(statsDAD.FP_zscore_STEPc6d)
        fprintf(fid, '\t%.2f\t\n', statsDAD.FP_zscore_STEPc6d(i,1));
    end
    fclose(fid);
    
    % ****************************
    curstep = 10;
    waitbar(curstep/steps);
    % ****************************
    
    
    
    
    
    
    % 5. (STEP c6e) - Calculate the Frequency Power (gamma: 32-49) of each channel
    fid = fopen('history.txt', 'at');
    fprintf(fid, '(STEP c6e) - Calculate the Frequency Power (gamma: 32-49) of each channel\n');
    fclose(fid);

    stepc6parm.filt_bandpass = [32 49];    % low and high frequencies for bandpass filter

    for i = 1:dad.NumOfsampledCHs
        stepc6parm.filtsignal_c6e{i,1} = PreprocessEEG(stepc6parm, DADmethod.vec_Acc_STEPc5{i,1});  % filter
        N = length(stepc6parm.filtsignal_c6e{i,1});
        f = (0:N-1)*statsDAD.Fs/(N-1);   % corresponding frequency vector
        stepc6parm.filtsignal_fft_c6e{i,1} = fft(stepc6parm.filtsignal_c6e{i,1}, N);  % Fourier transform
        stepc6parm.filtsignal_fft_gain_c6e{i,1} = abs(stepc6parm.filtsignal_fft_c6e{i,1}(1:round(length(f)/2)));  % gain (frequency response)
        DADmethod.FP_STEPc6e(i,1) = ((1/(2*pi))*(sum(stepc6parm.filtsignal_fft_gain_c6e{i,1})))/N;      % signal power
    end
    % -------------------------- Clear space
    stepc6parm = rmfield(stepc6parm, 'filtsignal_c6e');
    stepc6parm = rmfield(stepc6parm, 'filtsignal_fft_c6e');
    stepc6parm = rmfield(stepc6parm, 'filtsignal_fft_gain_c6e'); %#ok<NASGU>
    % --------------------------

    % A channel whose Frequency Power had a z-score of +/-3 would be deemed contaminated
    statsDAD.FP_zscore_STEPc6e = zscore(DADmethod.FP_STEPc6e); % Matlab's zscore: z = (data point-pop_mean)/pop_std

    fid = fopen('history.txt', 'at');
    for i = 1:length(statsDAD.FP_zscore_STEPc6e)
        fprintf(fid, '\t%.2f\t\n', statsDAD.FP_zscore_STEPc6e(i,1));
    end
    fclose(fid);



    fid = fopen('history.txt', 'at');
    fprintf(fid, 'Channel evaluation is COMPLETE...\n');
    fclose(fid);
    % -------------------------------------------------------------------------------------
    
    
    % ****************************
    curstep = 11;
    waitbar(curstep/steps);
    delete(h)
    % ****************************
    
    
    
    
    
    
    
    
    
    
    
    
    % -------------------------------------------------------------------------------------
    
    
    
    
    
    % ----------------------------------
    % Plot standard method data in subfigure 3 - artifact detection

    dad.data_eegch_dadart_full = DADmethod.vec_Acc_STEPlast{DADmethod.NumOfEXPs,1};
    
    dad.DADbadcounter_whole = statsDAD.rej_struct_whole{DADmethod.NumOfEXPs,1};
    
    r = length(dad.data_eegch_dadart_full)*(1/dad.data.samplingrate);
    dad.data_eegch_dadart_full_time = 0:(1/dad.data.samplingrate):(r - (1/dad.data.samplingrate));

    plot_data5;
    % ----------------------------------
    
    
    
    % Consolidate variables and save
    DADmethod.statsDAD = statsDAD;
    clear statsDAD
    
    save('DADmethod_whole.mat','DADmethod');
    
    % -------------------------- Clear space
    clear stepc6parm DADmethod temp
    % --------------------------
    
    
    save('dad.mat','dad');  % load dad.mat if it exist
    
end
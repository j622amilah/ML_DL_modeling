function runDAD_ep

% If dad.mat exists, load it
if exist('dad.mat', 'file') == 2
    load dad.mat
end

global dad

% Check if database is loaded
if dad.database == 0  % No database has been loaded
    errordlg('Load a Database (Import tab)...')
    
else
    % ****************************
    h = waitbar(0, 'Computing DAD (epochs)...');
    steps = 31;  % Total number of steps to complete the computation
    % ****************************
    
    dad.popup_splocValue = get(dad.popup_sploc, 'Value');
    temp.data_eegch = dad.data_eegchall{dad.popup_splocValue,1}; % Everytime the menu is changed, update the data vector
    
    % ----------------------------------
    % Calculate the DAD method 
    % ----------------------------------

    % (STEP 00) - Select the database 
    % Transform matrix of past data into cell array
    
    [a1, a2] = size(dad.database);      %#ok<ASGLU> % a2 is number of past data sets
    for EXPs = 1:a2
        DADmethod.database{EXPs,1} = dad.database(:,EXPs);
    end
    
    DADmethod.NumOfEXPs = a2+1; % Total number of datasets to compare at one time
    DADmethod.NumOfCHs = 1;     % Loading 1 spatial channel at a time
    
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
    statsDAD.filt_bandpass = [1 50];    % Low and high frequencies for bandpass filter
    statsDAD.Fs = dad.data.samplingrate;	 % 256


    for i = 1:DADmethod.NumOfEXPs
        DADmethod.filtsignal{i,1} = PreprocessEEG(statsDAD, DADmethod.database{i,1});
    end

    DADmethod.vec_Acc_STEPlast = DADmethod.filtsignal;

    dad.NumOfsampledCHs = DADmethod.NumOfEXPs*DADmethod.NumOfCHs;

    % -------------------------- Clear space
    DADmethod = rmfield(DADmethod, 'filtsignal');
    % -------------------------- 



    fid = fopen('history.txt', 'at');
    fprintf(fid, 'Channels will be evaluated before epochs...\n');
    fclose(fid);
    
    % ****************************
    curstep = 2;
    waitbar(curstep/steps);
    % ****************************




    % ---------------------------------- EVALUATE EPOCHS ----------------------------------


    % ooooooooooooooooooooooooooooo
    % Section: Epochs artifacts, Single-channel, single-epoch artifacts 
    % ooooooooooooooooooooooooooooo


    % (STEP e1) - Epoch each channel (cut the data in each channel every 10 seconds)
    fid = fopen('history.txt', 'at');
    fprintf(fid, '(STEP e1) - Epoch each channel (cut the data in each channel every 10 seconds)\n');
    fclose(fid);

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

    % -------------------------- Clear space
    clear i j
    DADmethod = rmfield(DADmethod, 'vec_Acc_STEPlast');
    % -------------------------- 
    
    % ****************************
    curstep = 3;
    waitbar(curstep/steps);
    % ****************************
    


    % -------------------------- 
    % Create a structure the same size as the original epochs to store the rejected information 
    % for statistics on which parts of the data are cleaned
    % -------------------------- 
    for i = 1:dad.NumOfsampledCHs
        statsDAD.rej_struct{i,1} = zeros(length(DADmethod.(sprintf('vec_epoch_Nsamp%d_STEPe1', i))), 1);
        statsDAD.rej_struct_gamma{i,1} = zeros(length(DADmethod.(sprintf('vec_epoch_Nsamp%d_STEPe1', i))), 1);
        statsDAD.rej_struct_beh{i,1} = zeros(length(DADmethod.(sprintf('vec_epoch_Nsamp%d_STEPe1', i))), 1);
    end






    % (STEP e2) - Baseline correct - zero mean the data and append "good" datasets to the epochs
    fid = fopen('history.txt', 'at');
    fprintf(fid, '(STEP e2) - Baseline correct - zero mean the data\n');
    fclose(fid);

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
    
    % ****************************
    curstep = 4;
    waitbar(curstep/steps);
    % ****************************
    
    
    % --------------------------
    fid = fopen('history.txt', 'at');
    fprintf(fid, 'Prevent detection bias: test the SNR of the data to determine which type of "good" datasets to use\n');
    fclose(fid);
    % --------------------------




    % --------------------------

    for i = 1:dad.NumOfsampledCHs
        for j = 1:length(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_STEPe2', i)))
            temp2.x{i,1}{j,1} = DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_STEPe2', i)){j,1};  % epoch I want to test

            % Randomly select one of the good data sets from the database
            b = 1;
            c = dad.NumOfsampledCHs-1;
            r = ceil(b + (c-b).*rand(1,1));
            temp2.y{i,1}{j,1} = DADmethod.database{r,1};  % the "good" epoch length signal

            N = length(temp2.x{i,1}{j,1});
            f = (0:N-1)*statsDAD.Fs/(N-1);   % corresponding frequency vector

            temp2.x_fft{i,1}{j,1} = fft(temp2.x{i,1}{j,1}, N);  % Fourier transform
            temp2.y_fft{i,1}{j,1} = fft(temp2.y{i,1}{j,1}, N);  % Fourier transform

            temp2.x_fft_gain{i,1}{j,1} = abs(temp2.x_fft{i,1}{j,1}(1:round(length(f)/2)));  % gain (frequency response)
            temp2.y_fft_gain{i,1}{j,1} = abs(temp2.y_fft{i,1}{j,1}(1:round(length(f)/2)));  % gain (frequency response)

            temp2.epoch_snr{i,1}(j,1) = (20*log10(norm(temp2.x_fft_gain{i,1}{j,1})./norm(temp2.x_fft_gain{i,1}{j,1}-temp2.y_fft_gain{i,1}{j,1})));
        end
    end

    for i = 1:dad.NumOfsampledCHs
        temp2.epoch_snr_zscore{i,1} = zscore(temp2.epoch_snr{i,1}); % Matlab's zscore: z = (data point-pop_mean)/pop_std
    end
    
    
    % ****************************
    curstep = 5;
    waitbar(curstep/steps);
    % ****************************

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

    % -------------------------- Clear space
    clear i j q a temp2 r b c
    % --------------------------

    statsDAD.Rej_snr_epoch = DADmethod.Rej_snr_epoch;
    
    % ****************************
    curstep = 6;
    waitbar(curstep/steps);
    % ****************************



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
    
    
    % ****************************
    curstep = 7;
    waitbar(curstep/steps);
    % ****************************



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


    % -------------------------- Clear space
    clear o1 o2 o1_start o1_end o2_start o2_end ii i j
    DADmethod = rmfield(DADmethod, 'pre_goodepochSet');
    for i = 1:dad.NumOfsampledCHs
        DADmethod = rmfield(DADmethod, sprintf('vec_epoch_Nsamp%d_STEPe1', i));
        DADmethod = rmfield(DADmethod, sprintf('vec_epoch_Nsamp%d_bc_STEPe2', i));
    end
    % --------------------------
    
    % ****************************
    curstep = 8;
    waitbar(curstep/steps);
    % ****************************
    
    
    
    
    
    
    % (STEP e3) - Variance within epoch
    fid = fopen('history.txt', 'at');
    fprintf(fid, '(STEP e3) - Variance within epoch\n');
    fclose(fid);

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
    
    % ****************************
    curstep = 9;
    waitbar(curstep/steps);
    % ****************************
    
    
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
    
    % ****************************
    curstep = 10;
    waitbar(curstep/steps);
    % ****************************
    
    
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

    % -------------------------- Clear space
    clear i k j q a counter
    for i = 1:dad.NumOfsampledCHs
        DADmethod = rmfield(DADmethod, sprintf('vec_epoch_Nsamp%d_bc_STEPe2a', i));
    end
    DADmethod = rmfield(DADmethod, 'epoch_var_STEPe3_FULL');
    DADmethod = rmfield(DADmethod, 'Rej_epoch_STEPe3_FULL');
    % --------------------------
    
    
    % ****************************
    curstep = 11;
    waitbar(curstep/steps);
    % ****************************




    % (STEP e4) - Calculate the Hurst exponent within the epoch
    fid = fopen('history.txt', 'at');
    fprintf(fid, '(STEP e4) - Calculate the Hurst exponent within epoch\n');
    fclose(fid);

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
    
    % ****************************
    curstep = 12;
    waitbar(curstep/steps);
    % ****************************
    
    
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
    
    % ****************************
    curstep = 13;
    waitbar(curstep/steps);
    % ****************************

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

    % -------------------------- Clear space
    clear i j q a
    for i = 1:dad.NumOfsampledCHs
        DADmethod = rmfield(DADmethod, sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe3_FULL', i));
    end
    DADmethod = rmfield(DADmethod, 'epoch_HE_STEPSe4_FULL');
    DADmethod = rmfield(DADmethod, 'Rej_epoch_STEPe4_FULL');
    % --------------------------
    
    % ****************************
    curstep = 14;
    waitbar(curstep/steps);
    % ****************************






    % (STEP e5) - Calculate the Frequency Power of each epoch
    fid = fopen('history.txt', 'at');
    fprintf(fid, '(STEP e5) - Calculate the Frequency Power of each epoch\n');
    fclose(fid);

    stepe5parm.order = statsDAD.order;
    stepe5parm.Fs = statsDAD.Fs;

    % --------------------------

    % 1. (STEP e5a) - Calculate the Frequency Power (delta: 1-3) of each epoch
    fid = fopen('history.txt', 'at');
    fprintf(fid, '(STEP e5a) - Calculate the Frequency Power (delta: 1-3) of each epoch\n');
    fclose(fid);

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
    
    % ****************************
    curstep = 15;
    waitbar(curstep/steps);
    % ****************************
    

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
    
    % ****************************
    curstep = 16;
    waitbar(curstep/steps);
    % ****************************


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
    
    % ****************************
    curstep = 17;
    waitbar(curstep/steps);
    % ****************************








    % 2. (STEP e5b) - Calculate the Frequency Power (theta: 4-7) of each epoch
    fid = fopen('history.txt', 'at');
    fprintf(fid, '(STEP e5b) - Calculate the Frequency Power (theta: 4-7) of each epoch\n');
    fclose(fid);

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
    
    % ****************************
    curstep = 18;
    waitbar(curstep/steps);
    % ****************************
    
    
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
    
    % ****************************
    curstep = 19;
    waitbar(curstep/steps);
    % ****************************
    
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
    
    % ****************************
    curstep = 20;
    waitbar(curstep/steps);
    % ****************************





    % 3. (STEP e5c) - Calculate the Frequency Power (alpha: 8-15) of each epoch
    fid = fopen('history.txt', 'at');
    fprintf(fid, '(STEP e5c) - Calculate the Frequency Power (alpha: 8-15) of each epoch\n');
    fclose(fid);

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
    
    % ****************************
    curstep = 21;
    waitbar(curstep/steps);
    % ****************************
    
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
                        % statsDAD.rej_struct{i,1}(a,1) = 52;    % Assign a 52 to all rejected epochs to denote (STEP e5c)
                        statsDAD.rej_struct_beh{i,1}(a,1) = 52;    % Assign a 52 to all rejected epochs to denote (STEP e5c)
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
    
    % ****************************
    curstep = 22;
    waitbar(curstep/steps);
    % ****************************






    % 4. (STEP e5d) - Calculate the Frequency Power (beta: 16-31) of each epoch
    fid = fopen('history.txt', 'at');
    fprintf(fid, '(STEP e5d) - Calculate the Frequency Power (beta: 16-31) of each epoch\n');
    fclose(fid);

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
    
    % ****************************
    curstep = 23;
    waitbar(curstep/steps);
    % ****************************
    
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
    
    % ****************************
    curstep = 24;
    waitbar(curstep/steps);
    % ****************************
    
    % --------------------------
    % Keep track of the rejected epochs and at which step they were rejected 0 indicates that the epoch is "good"
    % --------------------------
    for i = 1:dad.NumOfsampledCHs
        for k = 1:DADmethod.times2div(i,1)

            for j = 1:length(DADmethod.Rej_epoch_STEPe5d_FULL{i,1}{k,1})
                a = DADmethod.Rej_epoch_STEPe5d_FULL{i,1}{k,1}(j,1) + testCellsAmt*(k-1);

                if a < (statsDAD.Length_of_TOTtestcell(i,1) + 1)
                    if DADmethod.Rej_epoch_STEPe5d_FULL{i,1}{k,1}(j,1) < (testCellsAmt+1)
                        % statsDAD.rej_struct{i,1}(a,1) = 53;    % Assign a 53 to all rejected epochs to denote (STEP e5d)
                        statsDAD.rej_struct_beh{i,1}(a,1) = 53;    % Assign a 53 to all rejected epochs to denote (STEP e5d)
                    end
                end

            end

        end
    end

    % -------------------------- Clear space
    clear i j k q a N
    for i = 1:dad.NumOfsampledCHs      % 1:18
        DADmethod = rmfield(DADmethod, sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5c_FULL', i));
    end
    DADmethod = rmfield(DADmethod, 'FP_STEPe5d_FULL');
    DADmethod = rmfield(DADmethod, 'Rej_epoch_STEPe5d_FULL');
    % --------------------------
    
    % ****************************
    curstep = 25;
    waitbar(curstep/steps);
    % ****************************







    % 5. (STEP e5e) - Calculate the Frequency Power (gamma: 32-49) of each epoch
    fid = fopen('history.txt', 'at');
    fprintf(fid, '(STEP e5e) - Calculate the Frequency Power (gamma: 32-49) of each epoch\n');
    fclose(fid);

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
    
    % ****************************
    curstep = 26;
    waitbar(curstep/steps);
    % ****************************

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
                        statsDAD.rej_struct{i,1}(a,1) = 54;    % Assign a 54 to all rejected epochs to denote (STEP e5e)
                        statsDAD.rej_struct_gamma{i,1}(a,1) = 54;
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
    
    % ****************************
    curstep = 27;
    waitbar(curstep/steps);
    % ****************************
    
    
    
    
    
    
    
    
    % (STEP e6) - Peak within epoch
    fid = fopen('history.txt', 'at');
    fprintf(fid, '(STEP e6) - Peak within epoch\n');
    fclose(fid);
    
    for i = 1:dad.NumOfsampledCHs
        for k = 1:DADmethod.times2div(i,1)
            for j = 1:length(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5e_FULL', i)){k,1})

                sel = ( max(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5e_FULL', i)){k,1}{j,1}) - ...
                    min(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5e_FULL', i)){k,1}{j,1}) )/4;

                thresh = max(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5e_FULL', i)){k,1}{j,1})*0.55;

                [DADmethod.epoch_peakLoc_STEPe6_FULL{i,1}{k,1}{j,1}, DADmethod.epoch_peakMag_STEPe6_FULL{i,1}{k,1}{j,1}] = ...
                    peakfinder(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe5e_FULL', i)){k,1}{j,1}, sel, thresh);

                % Number of peaks present
                DADmethod.epoch_peaks_STEPe6_FULLall{i,1}{k,1}(j,1) = length(DADmethod.epoch_peakLoc_STEPe6_FULL{i,1}{k,1}{j,1});
                
                % Only interested in data segments that have 1-5 significant peaks present
                if DADmethod.epoch_peaks_STEPe6_FULLall{i,1}{k,1}(j,1) < 6
                    DADmethod.epoch_peaks_STEPe6_FULL{i,1}{k,1}(j,1) = DADmethod.epoch_peaks_STEPe6_FULLall{i,1}{k,1}(j,1);
                else
                    DADmethod.epoch_peaks_STEPe6_FULL{i,1}{k,1}(j,1) = 0;
                end
            end
        end
    end




    for i = 1:dad.NumOfsampledCHs
        for k = 1:DADmethod.times2div(i,1)
            statsDAD.epoch_peaks_zscore_STEPe6_FULL{i,1}{k,1} = zscore(DADmethod.epoch_peaks_STEPe6_FULL{i,1}{k,1}); 
            % Matlab's zscore: z = (data point-pop_mean)/pop_std
        end
    end







    
    % ****************************
    curstep = 28;
    waitbar(curstep/steps);
    % ****************************
    
    
    
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
    
    
    % ****************************
    curstep = 29;
    waitbar(curstep/steps);
    % ****************************
    
    
    
    
    
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
                            statsDAD.rej_struct_beh{i,1}(a,1) = 6;    % Assign a 6 to all rejected epochs to denote (STEP e6)
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
    statsDAD = rmfield(statsDAD, 'rej_struct_gamma');
    DADmethod = rmfield(DADmethod, 'epoch_peaks_STEPe6_FULLall');
    DADmethod = rmfield(DADmethod, 'epoch_peaks_STEPe6_FULL');
    DADmethod = rmfield(DADmethod, 'Rej_epoch_STEPe6_FULL');
    % --------------------------










    fid = fopen('history.txt', 'at');
    for i = 1:dad.NumOfsampledCHs      % 1:18

        fprintf(fid, 'NumOfsampledCHs: %d| \n', i);

        e = [3; 4; 50; 51; 52; 53; 54];
        for ii = 1:length(e)
            switch e(ii)
                case 3
                    fprintf(fid, 'variance: ');
                    for j = 1:length(statsDAD.rej_struct{i,1})
                        if statsDAD.rej_struct{i,1}(j,1) == e(ii)
                            fprintf(fid, '.%d.', j);
                        end
                    end

                case 4
                    fprintf(fid, 'hurst exponent: ');
                    for j = 1:length(statsDAD.rej_struct{i,1})
                        if statsDAD.rej_struct{i,1}(j,1) == e(ii)
                            fprintf(fid, '.%d.', j);
                        end
                    end

                case 50
                    fprintf(fid, 'freq delta 1-3Hz: ');
                    for j = 1:length(statsDAD.rej_struct{i,1})
                        if statsDAD.rej_struct{i,1}(j,1) == e(ii)
                            fprintf(fid, '.%d.', j);
                        end
                    end

                case 51
                    fprintf(fid, 'freq theta 4-7Hz: ');
                    for j = 1:length(statsDAD.rej_struct{i,1})
                        if statsDAD.rej_struct{i,1}(j,1) == e(ii)
                            fprintf(fid, '.%d.', j);
                        end
                    end

                case 52     % behavior
                    fprintf(fid, 'freq alpha 8-15Hz: ');
                    for j = 1:length(statsDAD.rej_struct_beh{i,1})
                        if statsDAD.rej_struct_beh{i,1}(j,1) == e(ii)
                            fprintf(fid, '.%d.', j);
                        end
                    end

                case 53     % behavior
                    fprintf(fid, 'freq beta 16-31Hz: ');
                    for j = 1:length(statsDAD.rej_struct_beh{i,1})
                        if statsDAD.rej_struct_beh{i,1}(j,1) == e(ii)
                            fprintf(fid, '.%d.', j);
                        end
                    end

                case 54
                    fprintf(fid, 'freq gamma 32-49Hz: ');
                    for j = 1:length(statsDAD.rej_struct{i,1})
                        if statsDAD.rej_struct{i,1}(j,1) == e(ii)
                            fprintf(fid, '.%d.', j);
                        end
                    end
                    
                case 6      % behavior: gamma and peaks
                    fprintf(fid, 'sig peaks: ');
                    for j = 1:length(statsDAD.rej_struct_beh{i,1})
                        if statsDAD.rej_struct_beh{i,1}(j,1) == e(ii)
                            fprintf(fid, '.%d.', j);
                        end
                    end
            end
            fprintf(fid, '\n');
        end
        fprintf(fid, '\n');
    end
    fclose(fid);



    fid = fopen('history.txt', 'at');
    fprintf(fid, 'Epoch evaluation is COMPLETE...\n');
    fclose(fid);
    
    % ****************************
    curstep = 30;
    waitbar(curstep/steps);
    % ****************************




    % --------------------------
    % Delete the appended cells from the goodepochSETs, so that the original data remains
    for i = 1:dad.NumOfsampledCHs
        for k = 1:DADmethod.times2div(i,1)
            for j = 1:DADmethod.Length_of_testcell{i,1}(k,1)
                DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEP7_FULL', i)){k,1}{j,1} = DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe6_FULL', i)){k,1}{j,1};
            end
        end
    end
    % --------------------------








    % --------------------------
    % Reduce the FULL length of the structure 
    q = 1;
    for i = 1:dad.NumOfsampledCHs
        for k = 1:DADmethod.times2div(i,1)
            for j = 1:DADmethod.Length_of_testcell{i,1}(k,1)

                if DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe6_FULL', i)){k,1}{j,1} ~= 0
                    DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEP7', i)){k,1}{q,1} = DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe6_FULL', i)){k,1}{j,1};
                    q = q + 1;
                end

            end
            q = 1;
        end
    end
    % --------------------------


    % -------------------------- Clear space
    clear i j k q ii e
    for i = 1:dad.NumOfsampledCHs
        DADmethod = rmfield(DADmethod, sprintf('vec_epoch_Nsamp%d_bc_Acc_STEPe6_FULL', i));
    end
    % --------------------------
    
    
    
    % ****************************
    curstep = 31;
    waitbar(curstep/steps);
    % ****************************


    % Reset the length of the number of sampled channels, in the case where all epochs within one channel are rejected
    counter = 1;
    for i = 1:dad.NumOfsampledCHs
        if isfield(DADmethod, (sprintf('vec_epoch_Nsamp%d_bc_Acc_STEP7', i))) ~= 0       % This field exists
            DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEP7_0', counter)) = DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEP7', i));
            counter = counter + 1;

            DADmethod = rmfield(DADmethod, sprintf('vec_epoch_Nsamp%d_bc_Acc_STEP7', i)); % Clear space
%         else
%             % Reject
%             % ..........
%             if isfield(statsDAD, 'testdata1_ch1') == 0 %go into loop if it does NOT exists
%                 if i == 8
%                     statsDAD.testdata1_ch1 = '1stpass_STEPe';
%                 end
%             else
%                 if isfield(statsDAD, 'testdata1_ch2') == 0 %go into loop if it does NOT exists
%                     if i == 15
%                         statsDAD.testdata1_ch2 = '1stpass_STEPe';
%                     end
%                 end
%             end
%             if isfield(statsDAD, 'testdata1_ch2') == 0 %go into loop if it does NOT exists
%                 if i == 16
%                     statsDAD.testdata1_ch2 = '1stpass_STEPe';
%                 end
%             end
%             % ..........
% 
%             fid = fopen('history.txt', 'at');
%             fprintf(fid, 'Rejected channels: %d\n', i);
%             fclose(fid);
        end
    end
    dad.NumOfsampledCHs = counter - 1;
    % --------------------------
    
    % ****************************
    curstep = 32;
    waitbar(curstep/steps);
    % ****************************



    % --------------------------
    fid = fopen('history.txt', 'at');
    fprintf(fid, 'DAD epoch - Transform data from epochs to channels\n');
    fclose(fid);

    holder = [];
    for i = 1:dad.NumOfsampledCHs
        for k = 1:length(DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEP7_FULL', i)))
            holder = cat(1, holder, DADmethod.(sprintf('vec_epoch_Nsamp%d_bc_Acc_STEP7_FULL', i)){k,1});
        end
        cclemon{i,1} = holder; %#ok<AGROW>
        holder = [];
    end

    holder = [];
    for i = 1:dad.NumOfsampledCHs
        for k = 1:length(cclemon{i,1})
            holder = cat(1, holder, cclemon{i,1}{k,1});
        end
        DADmethod.vec_Acc_STEPlast{i,1} = holder;
        holder = [];
    end
    % --------------------------


    % -------------------------- Clear space
    clear i k Length_of_cell testCellsAmt cclemon holder
    for i = 1:dad.NumOfsampledCHs
        DADmethod = rmfield(DADmethod, sprintf('vec_epoch_Nsamp%d_bc_Acc_STEP7_0', i));
        DADmethod = rmfield(DADmethod, sprintf('vec_epoch_Nsamp%d_bc_Acc_STEP7_FULL', i));
    end
    % --------------------------
    
    
    % ****************************
    curstep = 33;
    waitbar(curstep/steps);
    delete(h)
    % ****************************
    
    
    
    
    
    
    
    % -------------------------------------------------------------------------------------
    
    
    
    

    % ----------------------------------
    % Plot standard method data in subfigure 3 - artifact detection

    dad.data_eegch_dad10art_full = DADmethod.vec_Acc_STEPlast{DADmethod.NumOfEXPs,1};


    r = length(dad.data_eegch_dad10art_full)*(1/dad.data.samplingrate);
    dad.data_eegch_dad10art_full_time = 0:(1/dad.data.samplingrate):(r - (1/dad.data.samplingrate));

    dad.DADbadcounter = statsDAD.rej_struct{DADmethod.NumOfEXPs,1};

    plot_data3;
    % ----------------------------------


    % ----------------------------------
    % Plot standard method data in subfigure 4 - behavior detection
    
    % 1. Biting events are tagged during epoch evaluation: signigicant gamma frequency power z > 2, 5 or fewer significant peaks.
    % 2. Alpha events are tagged during both whole dataset and epoch evaluation of z > 2.5.
    
    dad.data_eegch_dad10beh_full = DADmethod.vec_Acc_STEPlast{DADmethod.NumOfEXPs,1};
    
    
    r = length(dad.data_eegch_dad10beh_full)*(1/dad.data.samplingrate);
    dad.data_eegch_dad10beh_full_time = 0:(1/dad.data.samplingrate):(r - (1/dad.data.samplingrate));
    
    dad.DADbadcounter_beh = statsDAD.rej_struct_beh{DADmethod.NumOfEXPs,1};
    
    plot_data4;
    % ----------------------------------
    
    
    
    % Consolidate variables and save
    DADmethod.statsDAD = statsDAD;
    clear statsDAD
    
    save('DADmethod_epochs.mat','DADmethod');

    % -------------------------- Clear space
    clear DADmethod temp
    % --------------------------
    
    
    save('dad.mat','dad');
    
end
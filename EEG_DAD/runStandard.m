function runStandard

% If dad.mat exists, load it
if exist('dad.mat', 'file') == 2
    load dad.mat
end

global dad

% ****************************
h = waitbar(0, 'Computing Standard (epochs)...');
steps = dad.constants.stand_STEP6;      % Total number of steps to complete the computation
% ****************************

dad.popup_splocValue = get(dad.popup_sploc, 'Value');
temp.data_eegch = dad.data_eegchall{dad.popup_splocValue,1}; % Everytime the menu is changed, update the data vector


% ----------------------------------
% Calculate the Standard method 
% ----------------------------------
% 1. Epoch the signal into 10 sec chunks
fid = fopen('history.txt', 'at');
fprintf(fid, 'Standard (STEP 1) - Epoch the signal into 10 sec chunks\n');
fclose(fid);

stand.EpochLength = 10;      % 10 second epochs
stand.EpochStart = 1;
stand.EpochEnd = (dad.data.samplingrate*stand.EpochLength) + 1;
j = 1;

while ( length(temp.data_eegch) >= stand.EpochEnd )
    dad.data_eegch_stand10{j,1} = temp.data_eegch(stand.EpochStart:stand.EpochEnd);
    stand.EpochStart = 1 + stand.EpochEnd;
    stand.EpochEnd = (dad.data.samplingrate*stand.EpochLength) + 1 + stand.EpochEnd;
    j = j + 1;
end

% ****************************
curstep = dad.constants.stand_STEP1;
waitbar(curstep/steps);
% ****************************


% 2. Choose a threshold for the standard method
fid = fopen('history.txt', 'at');
fprintf(fid, 'Standard (STEP 2) - Choose a threshold\n');
fclose(fid);

for i = 1:length(dad.data_eegch_stand10)     % length of 10 sec. chunks

    stand.mean(i,1) = mean(dad.data_eegch_stand10{i,1});

    stand.thresh_a(i,1) = max(dad.data_eegch_stand10{i,1});
    
    stand.data_eegch_stand10_thresh(i,1) = ((2/3))*(mean(stand.thresh_a(i,1)) - mean(stand.mean(i,1))) + mean(stand.mean(i,1));
end

% ****************************
curstep = dad.constants.stand_STEP2;
waitbar(curstep/steps);
% ****************************


% 3. Select cells based on threshold
fid = fopen('history.txt', 'at');
fprintf(fid, 'Standard (STEP 3) - Select cells based on threshold\n');
fclose(fid);

stand.badcount = 1; 
stand.goodcount = 1;
for i = 1:length(dad.data_eegch_stand10)     % length of 10 sec. chunks
    max_amp(i,1) = abs(max(dad.data_eegch_stand10{i,1})); %#ok<AGROW>
    if max_amp(i,1) > max(stand.data_eegch_stand10_thresh)
        
        % Reject
        dad.badcounter(stand.badcount, 1) = i;  % Identify the epochs that are exceed the threshhold
        
        % dad.data_eegch_stand10_r1{i,1} = zeros(2561,1);  % If you want to make the rejected sections equal to zero
        dad.data_eegch_stand10_r1{i,1} = dad.data_eegch_stand10{i,1};
        
        stand.badcount = stand.badcount + 1;
    else
        % Accept
        dad.data_eegch_stand10_r1{i,1} = dad.data_eegch_stand10{i,1};
        
        stand.data_eegch_stand10_r1_good{stand.goodcount,1} = dad.data_eegch_stand10{i,1};
        
        stand.goodcount = stand.goodcount + 1;
    end
end

% ****************************
curstep = dad.constants.stand_STEP3;
waitbar(curstep/steps);
% ****************************


% 4. Filter: Bandpass filter the data to remove frequencies greater than 40 Hz and lower than 1Hz
fid = fopen('history.txt', 'at');
fprintf(fid, 'Standard (STEP 4) - Filter: Bandpass filter the data to remove frequencies greater than 40 Hz and lower than 1Hz\n');
fclose(fid);

stand.order = 4;
stand.filt_bandpass = [1 40];    % low and high frequencies for bandpass filter
stand.Fs = dad.data.samplingrate;   % 256

for i = 1:length(dad.data_eegch_stand10_r1)
    dad.data_eegch_stand10_r2{i,1} = PreprocessEEG(stand, dad.data_eegch_stand10_r1{i,1});  % filter
end

% ****************************
curstep = dad.constants.stand_STEP4;
waitbar(curstep/steps);
% ****************************


% 5. Short results
for i = 1:length(dad.data_eegch_stand10_r2)
    if stand.goodcount == 1
        dad.data_eegch_stand10_result = 'allREJECT';
    else
        dad.data_eegch_stand10_result = 'someOK';
    end
end

% ****************************
curstep = dad.constants.stand_STEP5;
waitbar(curstep/steps);
% ****************************


% 6. Transform data from epochs to channels
fid = fopen('history.txt', 'at');
fprintf(fid, 'Standard (STEP 5) - Transform data from epochs to channels\n');
fclose(fid);

holder = [];
for k = 1:length(dad.data_eegch_stand10_r2)
    holder = cat(1, holder, dad.data_eegch_stand10_r2{k,1});
end
dad.data_eegch_stand10_r2full = holder;
clear i holder k cclemon

r = length(dad.data_eegch_stand10_r2full)*(1/dad.data.samplingrate);
dad.data_eegch_stand10_r2full_time = 0:(1/dad.data.samplingrate):(r - (1/dad.data.samplingrate));


dad.stand = stand;
clear stand j i max_amp temp


save('dad.mat','dad');

% ****************************
curstep = dad.constants.stand_STEP6;
waitbar(curstep/steps);
delete(h)
% ****************************



% 7. Plot standard method data in subfigure 2 - artifact detection
plot_data2;
% ----------------------------------

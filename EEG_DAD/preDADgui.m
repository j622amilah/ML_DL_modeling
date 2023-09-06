function preDADgui

%%

% Put data in the correct format
clear all
close all
clc

% ----------------------------------
% Make text files for each channel
% ----------------------------------
cd /Users/applee/Desktop/060414_IEEE_BiOCAS/Live_demonstration/DATA
load good


% Make a .txt file such that the columns are [time, C3]
X = [good.sub01_072513_1730_time', good.sub01_072513_1730_ch1'];
dlmwrite('sub1_data_C3.txt', X, 'delimiter', ' ', 'precision', '%10.6g');


% Make a .txt file such that the columns are [time, C4]
X = [good.sub01_072513_1730_time', good.sub01_072513_1730_ch2'];
dlmwrite('sub1_data_C4.txt', X, 'delimiter', ' ', 'precision', '%10.6g');


% Make a .txt file such that the columns are [time, C3, C4]
X = [good.sub01_072513_1730_time', good.sub01_072513_1730_ch1', good.sub01_072513_1730_ch2'];
dlmwrite('sub1_data_C3_C4.txt', X, 'delimiter', ' ', 'precision', '%10.6g');

%%

close all
clear all
clc

cd /Users/applee/Desktop/060414_IEEE_BiOCAS/Live_demonstration/DATA
load good   % good dataset: 7 experiments

NumofDatasets = 7; 

for EXPs = 1:NumofDatasets
    switch EXPs
        case 1  % Visually checked
            sub = '01';
            exp_date = '072513';
            tEXPs = '1055'; 
        case 2  % Visually checked
            sub = '01';
            exp_date = '072513';
            tEXPs = '1357';
        case 3  % Visually checked
            sub = '01';
            exp_date = '072513';
            tEXPs = '1434';
        case 4  % Visually checked
            sub = '01';
            exp_date = '072513';
            tEXPs = '1730';
        case 5  % Visually checked
            sub = '03';
            exp_date = '072613';
            tEXPs = '1136';
        case 6  % Visually checked
            sub = '03';
            exp_date = '072613';
            tEXPs = '1434';
        case 7  % Visually checked
            sub = '02';
            exp_date = '072613';
            tEXPs = '1720';

    end
    temp.time{EXPs,1} = good.(sprintf('sub%s_%s_%s_time', sub, exp_date, tEXPs));
    
    %temp.database_C3{EXPs,1} = good.(sprintf('sub%s_%s_%s_ch1', sub, exp_date, tEXPs));
    temp.database_C4{EXPs,1} = good.(sprintf('sub%s_%s_%s_ch2', sub, exp_date, tEXPs));
    
    %temp.database_length(EXPs,1) = length(temp.database_C3{EXPs,1});
    temp.database_length(EXPs,1) = length(temp.database_C4{EXPs,1});
end



clear good sub exp_date tEXPs EXPs



% 384150 is the number of data points from 25 minutes sampled at 256 Hz
% This length of data is required to perform the analysis
reqlength = 307320; %120*2561, (307320/256)/60 = 20 minutes 


for i = 1:NumofDatasets
    if temp.database_length(i,1) > (reqlength+1)
        %database_C3(:, i) = temp.database_C3{i,1}(1,1:reqlength)'; %#ok<AGROW>
        database_C4(:, i) = temp.database_C4{i,1}(1,1:reqlength)'; %#ok<AGROW>
    else
        diff = reqlength - temp.database_length(i,1);
        %database_C3(:,i) = [temp.database_C3{i,1}'; temp.database_C3{i,1}(1,1:diff)']; %#ok<AGROW>
        database_C4(:,i) = [temp.database_C4{i,1}'; temp.database_C4{i,1}(1,1:diff)']; %#ok<AGROW>
    end
end

% Save as a .txt file
%dlmwrite('database_C3.txt', database_C3, 'delimiter', ' ', 'precision', '%10.6g');
dlmwrite('database_C4.txt', database_C4, 'delimiter', ' ', 'precision', '%10.6g');

%%

% Make a .txt file such that the columns are [time, C3]
X = [good.sub01_072513_1730_time', good.sub01_072513_1730_ch1'];
dlmwrite('sub1_data_C3.txt', X, 'delimiter', ' ', 'precision', '%10.6g');


% Make a .txt file such that the columns are [time, C4]
X = [good.sub01_072513_1730_time', good.sub01_072513_1730_ch2'];
dlmwrite('sub1_data_C4.txt', X, 'delimiter', ' ', 'precision', '%10.6g');


% Make a .txt file such that the columns are [time, C3, C4]
X = [good.sub01_072513_1730_time', good.sub01_072513_1730_ch1', good.sub01_072513_1730_ch2'];
dlmwrite('sub1_data_C3_C4.txt', X, 'delimiter', ' ', 'precision', '%10.6g');



%%
% ---------------------------------------------------------
% ---------------------------------------------------------
% ---------------------------------------------------------
%%

% Bioradio dataset - read csv file and save as txt file
% My data for 10 mins resting with occassional movement/biting/increased alpha 

close all
clear all
clc

cd /home/dbi-data1/jamilah/Documents/Conferences/2014/060414_IEEE_BiOCAS/Live_demonstration/DATA/bioradio/

% ------------------
out.sub = 'sub1';

for q = 7:9
    out.chloc = [{'FC3_100614'}; {'C3_100614'}; {'FP1_100614'}; {'C3_101514'}; ...
            {'C3_101514_2'}; {'C3_101514_3'}; {'C3_101514_4'}; {'C3_101514_5'}; {'C3_101514_6'}];
    
    out.text_col1 = 'CH1';
    out.fileName = sprintf('%s_%s.csv', out.sub, out.chloc{q,1});
    [real_time, out.(sprintf('%s', out.chloc{q,1}))] = csvimport(out.fileName, 'columns', {'Real Time', out.text_col1});
    
    time2 = datevec(real_time);
    
    hour = time2(:,4);
    min = time2(:,5);
    sec = time2(:,6);

    % time string -> seconds
    [r, c] = size(time2); %#ok<NASGU>

    difftotsec(1) = 0;
    for i = 1:(r-1)

        sec_count = 0;

        if hour(i) == hour(i+1)
            difftotsec(i+1) = (sec(i+1) + min(i+1)*60) - (sec(i) + min(i)*60) + difftotsec(i);  %#ok<AGROW>
        else
            tempmin = min(i);
            tempsec = sec(i);
            while tempmin < 60
                sec_count = (60-tempsec) + sec_count;
                tempmin = tempmin + 1;
                tempsec = 0;
            end
            % falls out of loop when new hour is reached
            difftotsec(i+1) = (sec(i+1) + min(i+1)*60) + sec_count + difftotsec(i);   %#ok<AGROW>
        end
    end

    out.time = difftotsec';
    clear difftotsec sec min hour time1 time2 tempmin tempsec sec_count r c i date_time


    % Make a .txt file such that the columns are [time, eegchs]
    X = [out.time, out.(sprintf('%s', out.chloc{q,1}))];

    dlmwrite(sprintf('%s_rawdata_%s.txt', out.sub, out.chloc{q,1}), X, 'delimiter', ' ', 'precision', '%10.6g');
end

%%

% Bioradio dataset - visually see which datasets are least contaminated to select for database

clear all
close all
clc

addpath /home/dbi-data1/jamilah/Documents/Conferences/2014/060414_IEEE_BiOCAS/Live_demonstration

homedir = '/home/dbi-data1/jamilah/Documents/Conferences/2014/060414_IEEE_BiOCAS/Live_demonstration/DATA/bioradio';

fsize = 13;

for sub = 1:1
    for q = 2:2 %[2, 4, 6:9]
        out.chloc = [{'FC3_100614'}; {'C3_100614'}; {'FP1_100614'}; {'C3_101514'}; ...
            {'C3_101514_2'}; {'C3_101514_3'}; {'C3_101514_4'}; {'C3_101514_5'}; {'C3_101514_6'}];

        cd(homedir);
        M = dlmread(sprintf('sub%d_rawdata_%s.txt', sub, out.chloc{q,1}));
        time = M(:, 1);    % assume the 1st column is time
        data_eegchall = M(:,2);

        % Data statistics
        N = length(data_eegchall);
        samplingrate = round((N - 1) / (time(end) - time(1)));  % Fs = 256

        where_folder = sprintf('/home/dbi-data1/jamilah/Documents/Conferences/2014/060414_IEEE_BiOCAS/Live_demonstration/DATA/bioradio/sub%d_%s', sub, out.chloc{q,1});
        mkdir(where_folder);
        cd(where_folder);
        
        
        % Plot one second periods 
        a = 1;
        interval = samplingrate*2;    % 2 second long period = 512
        b = interval;
        
        Dset0 = data_eegchall;
        
        % Bandpass Filter [1 50]
        statsDAD.order = 4;
        statsDAD.filt_bandpass = [1 50];    % low and high frequencies for bandpass filter
        statsDAD.Fs = samplingrate;	 % 256
        
        Dset = PreprocessEEG(statsDAD, Dset0);
        numoftimes2view = floor(floor(length(Dset)/interval)/5);    % 5 is the number of subplots on one figure
        
        for i = 1:1 %numoftimes2view
            c1 = 0;
            c2 = 0;
            c3 = 1;
            
            figure('Visible', 'off')
            set(gcf, 'Position', [1 1 1680 1050]);
            
            subplot(2,3,1)
            plot(time(a:b, 1), Dset(a:b, 1), 'Color', [c1 c2 c3])
            title(sprintf('start=%d, end=%d', a, b))
            xlim([time(a,1) time(b,1)])
            intt = time(b,1)/4;
            xtick = [time(a,1);time(a,1)+intt;time(a,1)+2*intt;time(a,1)+3*intt;time(b,1)];
            xticklabel = num2str(xtick, '%.1f');
            set(gca, 'XTickLabelMode', 'manual', 'XMinorGrid', 'off', 'XTick', xtick, 'XTickLabel', xticklabel)
            
            subplot(2,3,2)
            plot(time((b+1:b+interval), 1), Dset((b+1:b+interval), 1), 'Color', [c1 c2 c3])
            title(sprintf('start=%d, end=%d', b+1, b+interval))
            xlim([time(b+1,1) time(b+interval,1)])
            xtick = [time(b+1,1);time(b+1,1)+intt;time(b+1,1)+2*intt;time(b+1,1)+3*intt;time(b+interval,1)];
            xticklabel = num2str(xtick, '%.1f');
            set(gca, 'XTickLabelMode', 'manual', 'XMinorGrid', 'off', 'XTick', xtick, 'XTickLabel', xticklabel)
            
            subplot(2,3,3)
            plot(time((b+interval+1:b+2*interval), 1), Dset((b+interval+1:b+2*interval), 1), 'Color', [c1 c2 c3])
            title(sprintf('start=%d, end=%d', b+interval+1, b+2*interval))
            xlim([time(b+interval+1,1) time(b+2*interval,1)])
            xtick = [time(b+interval+1,1);time(b+interval+1,1)+intt;time(b+interval+1,1)+2*intt;time(b+interval+1,1)+3*intt;time(b+2*interval,1)];
            xticklabel = num2str(xtick, '%.1f');
            set(gca, 'XTickLabelMode', 'manual', 'XMinorGrid', 'off', 'XTick', xtick, 'XTickLabel', xticklabel)
            
            subplot(2,3,4)
            plot(time((b+2*interval+1:b+3*interval), 1), Dset((b+2*interval+1:b+3*interval), 1), 'Color', [c1 c2 c3])
            title(sprintf('start=%d, end=%d', b+2*interval+1, b+3*interval))
            xlim([time(b+2*interval+1, 1) time(b+3*interval, 1)])
            xtick = [time(b+2*interval+1,1);time(b+2*interval+1,1)+intt;time(b+2*interval+1,1)+2*intt;time(b+2*interval+1,1)+3*intt;time(b+3*interval,1)];
            xticklabel = num2str(xtick, '%.1f');
            set(gca, 'XTickLabelMode', 'manual', 'XMinorGrid', 'off', 'XTick', xtick, 'XTickLabel', xticklabel)
            
            subplot(2,3,5)
            plot(time((b+3*interval+1:b+4*interval), 1), Dset((b+3*interval+1:b+4*interval), 1), 'Color', [c1 c2 c3])
            title(sprintf('start=%d, end=%d', b+3*interval+1, b+4*interval))
            xlim([time(b+3*interval+1, 1) time(b+4*interval, 1)])
            xtick = [time(b+3*interval+1,1);time(b+3*interval+1,1)+intt;time(b+3*interval+1,1)+2*intt;time(b+3*interval+1,1)+3*intt;time(b+4*interval,1)];
            xticklabel = num2str(xtick, '%.1f');
            set(gca, 'XTickLabelMode', 'manual', 'XMinorGrid', 'off', 'XTick', xtick, 'XTickLabel', xticklabel)
            xlabel('Time (s)', 'FontSize', fsize)
            
            subplot(2,3,6)
            plot(time((a:b+4*interval), 1), Dset((a:b+4*interval), 1), 'Color', [c1 c2 c3])
            title(sprintf('start=%d, end=%d', a, b+4*interval))
            xlim([time(a, 1) time(b+4*interval, 1)])
            intt = time(b+4*interval,1)/10;
            xtick = [time(a,1); time(a,1)+intt; time(a,1)+2*intt; time(a,1)+3*intt; time(b+4*interval,1)];
            xticklabel = num2str(xtick, '%.1f');
            set(gca, 'XTickLabelMode', 'manual', 'XMinorGrid', 'off', 'XTick', xtick, 'XTickLabel', xticklabel)
            
            print('-depsc2', '-r300', sprintf('figure%d', i))
            close all

            a = b+4*interval+1;
            b = b+5*interval;    % 2 second long period = 512
        end
        
    end
end

%%

close all
clear all
clc

cd /Users/applee/Desktop/060414_IEEE_BiOCAS/Live_demonstration/DATA
load good   % good dataset: 7 experiments

NumofDatasets = 7; 

for EXPs = 1:NumofDatasets
    switch EXPs
        case 1  % Visually checked
            sub = '01';
            exp_date = '072513';
            tEXPs = '1055'; 
        case 2  % Visually checked
            sub = '01';
            exp_date = '072513';
            tEXPs = '1357';
        case 3  % Visually checked
            sub = '01';
            exp_date = '072513';
            tEXPs = '1434';
        case 4  % Visually checked
            sub = '01';
            exp_date = '072513';
            tEXPs = '1730';
        case 5  % Visually checked
            sub = '03';
            exp_date = '072613';
            tEXPs = '1136';
        case 6  % Visually checked
            sub = '03';
            exp_date = '072613';
            tEXPs = '1434';
        case 7  % Visually checked
            sub = '02';
            exp_date = '072613';
            tEXPs = '1720';

    end
    temp.time{EXPs,1} = good.(sprintf('sub%s_%s_%s_time', sub, exp_date, tEXPs));
    
    %temp.database_C3{EXPs,1} = good.(sprintf('sub%s_%s_%s_ch1', sub, exp_date, tEXPs));
    temp.database_C4{EXPs,1} = good.(sprintf('sub%s_%s_%s_ch2', sub, exp_date, tEXPs));
    
    %temp.database_length(EXPs,1) = length(temp.database_C3{EXPs,1});
    temp.database_length(EXPs,1) = length(temp.database_C4{EXPs,1});
end



clear good sub exp_date tEXPs EXPs



% 384150 is the number of data points from 25 minutes sampled at 256 Hz
% This length of data is required to perform the analysis
reqlength = 307320; %120*2561, (307320/256)/60 = 20 minutes 


for i = 1:NumofDatasets
    if temp.database_length(i,1) > (reqlength+1)
        %database_C3(:, i) = temp.database_C3{i,1}(1,1:reqlength)'; %#ok<AGROW>
        database_C4(:, i) = temp.database_C4{i,1}(1,1:reqlength)'; %#ok<AGROW>
    else
        diff = reqlength - temp.database_length(i,1);
        %database_C3(:,i) = [temp.database_C3{i,1}'; temp.database_C3{i,1}(1,1:diff)']; %#ok<AGROW>
        database_C4(:,i) = [temp.database_C4{i,1}'; temp.database_C4{i,1}(1,1:diff)']; %#ok<AGROW>
    end
end

% Save as a .txt file
%dlmwrite('database_C3.txt', database_C3, 'delimiter', ' ', 'precision', '%10.6g');
dlmwrite('database_C4.txt', database_C4, 'delimiter', ' ', 'precision', '%10.6g');


%%



% Prepare public dataset (data vector and database) to test in DAD algorithm (http://mmspg.epfl.ch/cms/page-58322.html)

% Fp1, AF3, F7, F3, FC1, FC5, T7, C3, CP1, CP5, P7, P3, Pz, PO3, O1, Oz, O2, PO4, P4, P8, CP6, CP2, C4, T8, FC6, FC2, F4, F8, AF4, Fp2, Fz, Cz, MA1, MA2
% Each column corresponds to one temporal sample; the sampling rate is 2048 Hz.
% The data were recorded with a Biosemi Active Two system and are thus reference free.

% When working with the data please note that all experiments were performed under real-world conditions. 
% This means that the data might contain artifacts coming from eye-blinks, eye-movements, muscle-activity, etc (Hoffmann_2007_Efficient.pdf)

% Note: can make database from one subject's data or mutiple subject's data (1, 2, 6, 7) - Subjects 1 and 2 were able to perform simple, slow
% movements with their arms and hands but were unable to control other extremities. Spoken communication with subjects 1 and 2 was possible, 
% although both subjects suÆered from mild dysarthria. 

% Subjects 6 to 9 were PhD students recruited from our laboratory (all male, age 30). None of subjects 6 to 9 had known neurological defcits.


data_ref = data(1,:) - data(33,:); 




clear all
close all
clc

homedir = '/home/dbi-data1/jamilah/Documents/Conferences/2014/060414_IEEE_BiOCAS/Live_demonstration/DATA/public_dataset';
cd(homedir);

for sub = 1:1   %1:4
    
    for session = 1:1       %1:6
        for trial = 1:6
   

            % Visually look at some of the data
            cd(sprintf('%s/subject%d/session%d', homedir, sub, session));

            switch session
                case 1
                    switch trial
                        case 1
                            load eeg_200605191428_epochs.mat
                        case 2
                            
                        case 3
                            
                        case 4
                            
                        case 5 
                            
                        case 6
                        
                    end
                    
            end

            % loads data, events, stimuli, target, targets_counted
            % Fp1, AF3, F7, F3, FC1, FC5, T7, C3, CP1, CP5, P7, P3, Pz, PO3, O1, Oz, O2, PO4, P4, P8, CP6, CP2, C4, T8, FC6, FC2, F4, F8, AF4, Fp2, Fz, Cz, MA1, MA2
            
            out.chloc = [{'FC3'}; {'C3'}; {'FP1'}];
            
            time = M(:, 1);    % assume the 1st column is time
            data_eegchall = M(:,2);

            % Data statistics
            N = length(data_eegchall);
            samplingrate = round((N - 1) / (time(end) - time(1)));  % Fs = 256

            where_folder = sprintf('/home/dbi-data1/jamilah/Documents/Conferences/2014/060414_IEEE_BiOCAS/Live_demonstration/DATA/bioradio/sub%d_%s', sub, out.chloc{q,1});
            mkdir(where_folder);
            cd(where_folder);

            EpochLength = 10;      % 10 second epochs
            EpochStart = 1;
            EpochEnd = (samplingrate*EpochLength) + 1;
            j = 1;

            while ( length(data_eegchall) >= EpochEnd )
                data_eegch10{j,1} = data_eegchall(EpochStart:EpochEnd); %#ok<AGROW>

                figure('Visible', 'off')
                plot(data_eegch10{j,1}, '-b')
                print('-depsc2', '-r300', sprintf('figure%d', j))
                close all

                EpochStart = 1 + EpochEnd;
                EpochEnd = (samplingrate*EpochLength) + 1 + EpochEnd;
                j = j + 1;
            end
            
            
        end
    end
end



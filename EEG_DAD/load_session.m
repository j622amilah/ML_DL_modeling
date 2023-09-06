function load_session

if exist('dad.mat', 'file') == 2
    load('dad.mat');
else
    msgbox('dad.mat file does not exist.  Please load a dataset and database.', 'Help message', 'help')
end



if exist('dad.data_eegchall', 'var') == 1
    plot_data1;
    
    if exist('dad.data_eegch_stand10_r2full', 'var') == 1
        plot_data2;
    else
    end
    
    if exist('dad.data_eegch_dad10art_full', 'var') == 1
        plot_data3;
    else
    end
    
    if exist('dad.data_eegch_dad10beh_full', 'var') == 1
        plot_data4;
    else
    end
    
else
    msgbox('No data found in dad.mat. Please load a dataset and database.', 'Help message', 'help')
end




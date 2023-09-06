function plot_data1

global dad

if length(dad.splocation) == dad.numofrawdata
    % Subfigure 1 - Display raw data
    axes(dad.gui.overview.ax1);
    cla;
    hold on
    
    dad.gui.overview.ch1 = plot(dad.data.time, dad.data_eegchall{dad.popup_splocValue,1}, 'ButtonDownFcn', 'dad_click(1)', ...
        'Color', dad.gui.darkblue, 'LineWidth', 1);
    
    set(dad.gui.overview.ax1, 'XLim', [0, dad.data.time(end)], 'YLim', [-2 2], 'Color', dad.gui.lightgray)
    
    
else
    errordlg('The number of rawdata columns does not equal the number of entered EEG channels.  Reload data.')
end

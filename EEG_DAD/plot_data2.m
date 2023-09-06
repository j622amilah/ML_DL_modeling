function plot_data2

global dad

% Subfigure 2 - comparison algorithm output (overlay detected artifact areas)
axes(dad.gui.overview.ax2);
cla;
hold on

dad.gui.overview.ch2 = plot(dad.data_eegch_stand10_r2full_time, dad.data_eegch_stand10_r2full, 'ButtonDownFcn', 'dad_click(2)', 'Color', dad.gui.darkblue, 'LineWidth', 1);


% Plot shaded areas over places where artifacts occur
for i = 1:length(dad.badcounter)
    
    u1 = round((dad.badcounter(i,1) - 1)*2561);
    x1 = u1/dad.data.samplingrate;
    
    x2 = x1 + 10;
    dad.gui.overview.ch2 = fill([x1 x2 x2 x1], [-2 -2 2 2], dad.gui.lightblue, 'LineStyle', ':', 'FaceAlpha', 0.8, 'ButtonDownFcn', 'dad_click(2)');
end


clear r i x1 x2


% ------- IF YOU WANT THE Y-AXIS TO REMAIN [-2 2] -------
set(dad.gui.overview.ax2, 'XLim', [0 dad.data_eegch_stand10_r2full_time(end)], 'YLim', [-2 2], 'Color', dad.gui.lightgray)
% -----------------------------------------------------------
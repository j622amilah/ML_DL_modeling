function save_session

global dad

% -----------------
% Save variable so you can load your session later
% -----------------
if exist('dad.mat', 'file') == 2
    load('dad.mat');
else
end


if exist('DADmethod_epochs.mat', 'file') == 2
    load('DADmethod.mat');
    dad.DADmethod_epochs = DADmethod;
    delete('DADmethod.mat');
else
    dad.DADmethod_epochs = DADmethod;
end


if exist('DADmethod_whole.mat', 'file') == 2
    load('DADmethod.mat');
    dad.DADmethod_whole = DADmethod;
    delete('DADmethod.mat');
else
    dad.DADmethod_whole = DADmethod;
end

save('dad.mat','dad');





% -----------------
% Save an image of the screen
% -----------------
set(dad.gui.fig_main, 'PaperType', 'A4', 'PaperOrientation', 'landscape');       % 'portrait'  'landscape'
set(gcf, 'PaperPositionMode', 'manual', 'PaperUnits', 'normalized', 'Paperposition', [-0.5 -0.5 1.0 1.0]);

printfontsize = 8;
% Subfigure 1 - displays raw data
set(dad.gui.overview.ax1, 'YLim', [-2 2], 'Color', [0.9 0.9 0.9], 'FontSize', printfontsize);
set(get(dad.gui.overview.ax1, 'YLabel'), 'String', 'EEG raw data [V]', 'FontSize', printfontsize)


% Subfigure 2 - comparison algorithm output (overlay detected artifact areas)
set(dad.gui.overview.ax2, 'YLim', [-2 2], 'Color', [0.9 0.9 0.9], 'FontSize', printfontsize);
set(get(dad.gui.overview.ax2, 'YLabel'), 'String', 'Standard method - art [V]', 'FontSize', printfontsize)


% Subfigure 3 - DAD algorithm output (overlay detected artifact areas)
set(dad.gui.overview.ax3, 'YLim', [-2 2], 'Color', [0.9 0.9 0.9], 'FontSize', printfontsize);
set(get(dad.gui.overview.ax3, 'YLabel'), 'String', 'DAD method - art [V]', 'FontSize', printfontsize)


% Subfigure 4 - DAD algorithm output (overlay detected behavioral tag areas)
set(dad.gui.overview.ax4, 'YLim', [-2 2], 'Color', [0.9 0.9 0.9], 'FontSize', printfontsize);
set(get(dad.gui.overview.ax4, 'YLabel'), 'String', 'DAD method - beh [V]', 'FontSize', printfontsize)
set(get(dad.gui.overview.ax4, 'XLabel'), 'String', 'Time [sec]', 'FontSize', printfontsize)

print('-depsc2', '-r300', 'output_DADfig')
% -----------------


% -----------------
% Return settings back
set(dad.gui.overview.ax1, 'YLim', [-2 2], 'Color', [0.9 0.9 0.9], 'FontSize', 14);
set(get(dad.gui.overview.ax1, 'YLabel'), 'String', {'Raw data'; 'EEG [V]'}, 'FontSize', 14)

set(dad.gui.overview.ax2, 'YLim', [-2 2], 'Color', [0.9 0.9 0.9], 'FontSize', 14);
set(get(dad.gui.overview.ax2, 'YLabel'), 'String', {'Standard method'; 'EEG w/ artifact tags [V]'}, 'FontSize', 14)

set(dad.gui.overview.ax3, 'YLim', [-2 2], 'Color', [0.9 0.9 0.9], 'FontSize', 14);
set(get(dad.gui.overview.ax3, 'YLabel'), 'String', {'DAD method'; 'EEG w/ artifact tags [V]'}, 'FontSize', 14)

set(dad.gui.overview.ax4, 'YLim', [-2 2], 'Color', [0.9 0.9 0.9], 'FontSize', 14);
set(get(dad.gui.overview.ax4, 'YLabel'), 'String', {'DAD method'; 'EEG w/ behavior tags [V]'}, 'FontSize', 14)
set(get(dad.gui.overview.ax4, 'XLabel'), 'String', 'Time [sec]', 'FontSize', 14)
% -----------------

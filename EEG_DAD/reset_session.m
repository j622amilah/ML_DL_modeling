function reset_session

global dad


set(dad.popup_sploc, 'String', 'NA');
set(dad.popup_sploc, 'Value', 1);

% -----------------
% Subfigure 1 - displays raw data
axes(dad.gui.overview.ax1);
cla;
set(dad.gui.overview.ax1, 'XLim', [0 60], 'YLim', [-2 2], 'Color', [0.9 0.9 0.9], 'FontSize', 14);
set(get(dad.gui.overview.ax1, 'YLabel'), 'String', {'Raw data'; 'EEG [V]'}, 'FontSize', 14)

dad.gui.rangeview1.start = 0;
dad.gui.rangeview1.range = dad.widthofsignal;
set(dad.gui.rangeview1.edit_start, 'String', dad.gui.rangeview1.start);
set(dad.gui.rangeview1.edit_range, 'String', dad.gui.rangeview1.range);
set(dad.gui.rangeview1.edit_end, 'String',  dad.gui.rangeview1.start + dad.gui.rangeview1.range);
set(dad.gui.rangeview1.slider, 'Position', [0.05 0.96 0.87 0.02]);
% -----------------


% -----------------
% Subfigure 2 - comparison algorithm output (overlay detected artifact areas)
axes(dad.gui.overview.ax2);
cla;
set(dad.gui.overview.ax2, 'XLim', [0 60], 'YLim', [-2 2], 'Color', [0.9 0.9 0.9], 'FontSize', 14);
set(get(dad.gui.overview.ax2, 'YLabel'), 'String', {'Standard method'; 'EEG w/ artifact tags [V]'}, 'FontSize', 14)

dad.gui.rangeview2.start = 0;
dad.gui.rangeview2.range = dad.widthofsignal;
set(dad.gui.rangeview2.edit_start, 'String', dad.gui.rangeview2.start);
set(dad.gui.rangeview2.edit_range, 'String', dad.gui.rangeview2.range);
set(dad.gui.rangeview2.edit_end, 'String',  dad.gui.rangeview2.start + dad.gui.rangeview2.range);
set(dad.gui.rangeview2.slider, 'Position', [0.05 0.72 0.87 0.02]);
% -----------------


% -----------------
% Subfigure 3 - DAD algorithm output (overlay detected artifact areas)
axes(dad.gui.overview.ax3);
cla;
set(dad.gui.overview.ax3, 'XLim', [0 60], 'YLim', [-2 2], 'Color', [0.9 0.9 0.9], 'FontSize', 14);
set(get(dad.gui.overview.ax3, 'YLabel'), 'String', {'DAD method'; 'EEG w/ artifact tags [V]'}, 'FontSize', 14)

dad.gui.rangeview3.start = 0;
dad.gui.rangeview3.range = dad.widthofsignal;
set(dad.gui.rangeview3.edit_start, 'String', dad.gui.rangeview3.start);
set(dad.gui.rangeview3.edit_range, 'String', dad.gui.rangeview3.range);
set(dad.gui.rangeview3.edit_end, 'String',  dad.gui.rangeview3.start + dad.gui.rangeview3.range);
set(dad.gui.rangeview3.slider, 'Position', [0.05 0.48 0.87 0.02]);
% -----------------


% -----------------
% Subfigure 4 - DAD algorithm output (overlay detected behavioral tag areas)
axes(dad.gui.overview.ax4);
cla;
set(dad.gui.overview.ax4, 'XLim', [0 60], 'YLim', [-2 2], 'Color', [0.9 0.9 0.9], 'FontSize', 14);
set(get(dad.gui.overview.ax4, 'YLabel'), 'String', {'DAD method'; 'EEG w/ behavior tags [V]'}, 'FontSize', 14)

dad.gui.rangeview4.start = 0;
dad.gui.rangeview4.range = dad.widthofsignal;
set(dad.gui.rangeview4.edit_start, 'String', dad.gui.rangeview4.start);
set(dad.gui.rangeview4.edit_range, 'String', dad.gui.rangeview4.range);
set(dad.gui.rangeview4.edit_end, 'String',  dad.gui.rangeview4.start + dad.gui.rangeview4.range);
set(dad.gui.rangeview4.slider, 'Position', [0.05 0.24 0.87 0.02]);
% -----------------



clear all
clc


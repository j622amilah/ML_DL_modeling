function DADgui

clc
close all
clear all


% If history.txt exists, delete it
if exist('history.txt', 'file') == 2
    delete history.txt
else
end

% If dad.mat exists, delete it
if exist('dad.mat', 'file') == 2
    delete dad.mat
else
end

global dad

dad.intern.name = 'DAD';
dad.intern.version = 0.1;
versiontxt = num2str(dad.intern.version, '%3.2f');
dad.intern.versiontxt = ['Version', versiontxt(1:3), ' ', versiontxt(4:end)];
dad.intern.version_datestr = '2014-09-13';

% Add all subdirectories to Matlab path
file = which('DADgui.m');
if isempty(file)
    errordlg('Can not find the DADgui.m file, please change to DADgui.m directory.');
    return;
end

dad.intern.install_dir = fileparts(file);

addpath(genpath(dad.intern.install_dir));


% -----------------
% Load preset/initial conditons
dadpresettings;

% -----------------



% -----------------
% Load the logo
dadlogo(1);
pause(1);
delete(dad.gui.fig_logo);   % delete figure
% -----------------


% -----------------
% Background GUI
dad.gui.fig_main = figure('Units', 'normalized', 'Position', [0 0.03 1 0.92], 'Name', [dad.intern.name, ' ', dad.intern.versiontxt], ...
    'MenuBar', 'none', 'NumberTitle', 'off', 'Color', dad.gui.gray_color, 'CloseRequestFcn', 'exitdad');
% -----------------



% -----------------
% Load data: this creates a menu at the top of the figure where you can click on to search for data
dad.gui.menu.menu_1  = uimenu(dad.gui.fig_main, 'Label', 'File');
dad.gui.menu.menu_1b  = uimenu(dad.gui.menu.menu_1, 'Label', 'Load', 'Callback', 'load_session;');
dad.gui.menu.menu_1b1  = uimenu(dad.gui.menu.menu_1, 'Label', 'Save', 'Callback', 'save_session;');
dad.gui.menu.menu_1b2  = uimenu(dad.gui.menu.menu_1, 'Label', 'Reset', 'Callback', 'reset_session;');
dad.gui.menu.menu_1b3  = uimenu(dad.gui.menu.menu_1, 'Label', 'Exit', 'Callback', 'exitdad;');

dad.gui.menu.menu_2  = uimenu(dad.gui.fig_main, 'Label', 'Import');
dad.gui.menu.menu_2b  = uimenu(dad.gui.menu.menu_2, 'Label', 'Raw EEG data');
dad.gui.menu.menu_2bb = uimenu(dad.gui.menu.menu_2b, 'Label', 'Text (*.txt) [time, ch1]', 'Callback', 'import_data(''text'');');

dad.gui.menu.menu_2b1  = uimenu(dad.gui.menu.menu_2, 'Label', 'DAD database');
dad.gui.menu.menu_2bb1 = uimenu(dad.gui.menu.menu_2b1, 'Label', 'Text (*.txt) [set1, set2, .. , set8]', 'Callback', 'import_database(''text'');');
% dad.gui.menu.menu_2bb2 = uimenu(dad.gui.menu.menu_2b1,  'Label', 'Matlab (*.mat) [set1, set2, .. , set8]', 'Callback', 'import_database(''mat'');', 'Separator','on');

dad.gui.menu.menu_3  = uimenu(dad.gui.fig_main, 'Label', 'Compute');
dad.gui.menu.menu_3b  = uimenu(dad.gui.menu.menu_3, 'Label', 'Standard (epochs)', 'Callback', 'runStandard;');
dad.gui.menu.menu_3b1  = uimenu(dad.gui.menu.menu_3, 'Label', 'DAD (whole)', 'Callback', 'runDAD_ch;');
dad.gui.menu.menu_3b2  = uimenu(dad.gui.menu.menu_3, 'Label', 'DAD (epochs)', 'Callback', 'runDAD_ep;');
dad.gui.menu.menu_3b3  = uimenu(dad.gui.menu.menu_3, 'Label', 'DAD (whole, epochs)', 'Callback', 'runDAD_ch_ep;');
dad.gui.menu.menu_3b4  = uimenu(dad.gui.menu.menu_3, 'Label', 'Standard (epochs) & DAD (whole, epochs)', 'Callback', 'runall;');

dad.gui.menu.menu_4  = uimenu(dad.gui.fig_main, 'Label', 'Help');
dad.gui.menu.menu_4b  = uimenu(dad.gui.menu.menu_4, 'Label', 'About', 'Callback', 'dadlogo(2);');
% -----------------


dad.widthofsignal = 30;


% -----------------
% Subfigure 1 - displays raw data
dy1 = 0.78;
dad.gui.overview.ax1 = axes('Units', 'normalized', 'Position', [0.05 dy1 0.87 0.18], 'ButtonDownFcn', 'dad_click(1)');
set(dad.gui.overview.ax1, 'XLim', [0 60], 'YLim', [-2 2], 'Color', [0.9 0.9 0.9], 'FontSize', 14);
set(get(dad.gui.overview.ax1, 'YLabel'), 'String', {'Raw data'; 'EEG [V]'}, 'FontSize', 14)



dad.gui.rangeview1.start = 0;
dad.gui.rangeview1.range = dad.widthofsignal;
dad.gui.rangeview1.edit_start = uicontrol('Units', 'normalized', 'Style', 'edit', 'Position', [0.921 0.956 0.04 0.025], 'String', dad.gui.rangeview1.start, 'HorizontalAlignment', 'center', 'Callback', 'edits1(1)');
dad.gui.rangeview1.edit_range = uicontrol('Units', 'normalized', 'Style', 'edit', 'Position', [0.921 0.956 0.04 0.025], 'String', dad.gui.rangeview1.range, 'HorizontalAlignment', 'center', 'Callback', 'edits1(1)');
dad.gui.rangeview1.edit_end = uicontrol('Units', 'normalized', 'Style', 'edit', 'Position', [0.921 0.956 0.04 0.025], 'String',  dad.gui.rangeview1.start + dad.gui.rangeview1.range, 'HorizontalAlignment', 'center', 'Callback', 'edits1(2)');

dad.gui.rangeview1.slider = uicontrol('Style', 'Slider', 'Units', 'normalized', 'Position', [0.05 0.96 0.87 0.02], 'Min', 0, 'Max', 1, 'SliderStep', [0.01 0.1], 'Callback', 'edits1(3)');

dad.popup_splocValue = 1;
dad.popup_sploc = uicontrol(dad.gui.fig_main, 'Style', 'popupmenu', 'String', 'NA', 'Value', dad.popup_splocValue, 'Position', [1140 640 80 20], 'Callback', 'edits1(4)');
% -----------------


% -----------------
% Subfigure 2 - comparison algorithm output (overlay detected artifact areas)
dy2 = 0.54;
dad.gui.overview.ax2 = axes('Units', 'normalized', 'Position', [0.05 dy2 0.87 0.18], 'ButtonDownFcn', 'dad_click(2)');
set(dad.gui.overview.ax2, 'XLim', [0 60], 'YLim', [-2 2], 'Color', [0.9 0.9 0.9], 'FontSize', 14);
set(get(dad.gui.overview.ax2, 'YLabel'), 'String', {'Standard method'; 'EEG w/ artifact tags [V]'}, 'FontSize', 14)
% set(get(dad.gui.overview.ax2, 'XLabel'), 'String', 'Time [sec]', 'FontSize', 14)

dad.gui.rangeview2.start = 0;
dad.gui.rangeview2.range = dad.widthofsignal;
dad.gui.rangeview2.edit_start = uicontrol('Units', 'normalized', 'Style', 'edit', 'Position', [0.921 0.716 0.04 0.025], 'String', dad.gui.rangeview2.start, 'HorizontalAlignment', 'center', 'Callback', 'edits2(1)');
dad.gui.rangeview2.edit_range = uicontrol('Units', 'normalized', 'Style', 'edit', 'Position', [0.921 0.716 0.04 0.025], 'String', dad.gui.rangeview2.range, 'HorizontalAlignment', 'center', 'Callback', 'edits2(1)');
dad.gui.rangeview2.edit_end = uicontrol('Units', 'normalized', 'Style', 'edit', 'Position', [0.921 0.716 0.04 0.025], 'String',  dad.gui.rangeview2.start + dad.gui.rangeview2.range, 'HorizontalAlignment', 'center', 'Callback', 'edits2(2)');
dad.gui.rangeview2.slider = uicontrol('Style', 'Slider', 'Units', 'normalized', 'Position', [0.05 0.72 0.87 0.02], 'Min', 0, 'Max', 1, 'SliderStep', [0.01 0.1], 'Callback', 'edits2(3)');
% -----------------


% -----------------
% Subfigure 3 - DAD algorithm output (overlay detected artifact areas)
dy3 = 0.3;
dad.gui.overview.ax3 = axes('Units', 'normalized', 'Position', [0.05 dy3 0.87 0.18], 'ButtonDownFcn', 'dad_click(3)');
set(dad.gui.overview.ax3, 'XLim', [0 60], 'YLim', [-2 2], 'Color', [0.9 0.9 0.9], 'FontSize', 14);
set(get(dad.gui.overview.ax3, 'YLabel'), 'String', {'DAD method'; 'EEG w/ artifact tags [V]'}, 'FontSize', 14)
% set(get(dad.gui.overview.ax3, 'XLabel'), 'String', 'Time [sec]', 'FontSize', 14)

dad.gui.rangeview3.start = 0;
dad.gui.rangeview3.range = dad.widthofsignal;
dad.gui.rangeview3.edit_start = uicontrol('Units', 'normalized', 'Style', 'edit', 'Position', [0.921 0.476 0.04 0.025], 'String', dad.gui.rangeview3.start, 'HorizontalAlignment', 'center', 'Callback', 'edits3(1)');
dad.gui.rangeview3.edit_range = uicontrol('Units', 'normalized', 'Style', 'edit', 'Position', [0.921 0.476 0.04 0.025], 'String', dad.gui.rangeview3.range, 'HorizontalAlignment', 'center', 'Callback', 'edits3(1)');
dad.gui.rangeview3.edit_end = uicontrol('Units', 'normalized', 'Style', 'edit', 'Position', [0.921 0.476 0.04 0.025], 'String',  dad.gui.rangeview3.start + dad.gui.rangeview3.range, 'HorizontalAlignment', 'center', 'Callback', 'edits3(2)');
dad.gui.rangeview3.slider = uicontrol('Style', 'Slider', 'Units', 'normalized', 'Position', [0.05 0.48 0.87 0.02], 'Min', 0, 'Max', 1, 'SliderStep', [0.01 0.1], 'Callback', 'edits3(3)');
% -----------------


% -----------------
% Subfigure 4 - DAD algorithm output (overlay detected behavioral tag areas)
dy4 = 0.06;
dad.gui.overview.ax4 = axes('Units', 'normalized', 'Position', [0.05 dy4 0.87 0.18], 'ButtonDownFcn', 'dad_click(4)');
set(dad.gui.overview.ax4, 'XLim', [0 60], 'YLim', [-2 2], 'Color', [0.9 0.9 0.9], 'FontSize', 14);
set(get(dad.gui.overview.ax4, 'YLabel'), 'String', {'DAD method'; 'EEG w/ behavior tags [V]'}, 'FontSize', 14)
set(get(dad.gui.overview.ax4, 'XLabel'), 'String', 'Time [sec]', 'FontSize', 14)

dad.gui.rangeview4.start = 0;
dad.gui.rangeview4.range = dad.widthofsignal;
dad.gui.rangeview4.edit_start = uicontrol('Units', 'normalized', 'Style', 'edit', 'Position', [0.921 0.236 0.04 0.025], 'String', dad.gui.rangeview4.start, 'HorizontalAlignment', 'center', 'Callback', 'edits4(1)');
dad.gui.rangeview4.edit_range = uicontrol('Units', 'normalized', 'Style', 'edit', 'Position', [0.921 0.236 0.04 0.025], 'String', dad.gui.rangeview4.range, 'HorizontalAlignment', 'center', 'Callback', 'edits4(1)');
dad.gui.rangeview4.edit_end = uicontrol('Units', 'normalized', 'Style', 'edit', 'Position', [0.921 0.236 0.04 0.025], 'String',  dad.gui.rangeview4.start + dad.gui.rangeview4.range, 'HorizontalAlignment', 'center', 'Callback', 'edits4(2)');
dad.gui.rangeview4.slider = uicontrol('Style', 'Slider', 'Units', 'normalized', 'Position', [0.05 0.24 0.87 0.02], 'Min', 0, 'Max', 1, 'SliderStep', [0.01 0.1], 'Callback', 'edits4(3)');
% -----------------


end



% -----------------------------------------

function dadpresettings

global dad

dad.gui.lightgray = [229.50 229.50 229.50]./255;    % color around y-axis
dad.gui.gray_color = [204 204 204]./255;    % color of figure
dad.gui.darkblue = [0 0 153]./255;          % color of EEG signal

% Artifact detection tags: color of shaded artifact locations
dad.gui.lightblue = [204 229 225]./255;     % Standard method ONLY, variance DAD method (ch & epoch)
dad.gui.lightgreen = [229 255 204]./255;    % median DAD method (ch)
dad.gui.lightorange = [255 229 204]./255;   % Mean Corr DAD method (ch) **
dad.gui.lightred = [255 204 204]./255;      % Hurst DAD method (ch & epoch)
dad.gui.darkgray = [128 128 128]./255;      % delta frequency DAD method (ch & epoch)
dad.gui.olivegreen = [153 153 0]./255;      % theta frequency DAD method (ch & epoch)
dad.gui.yellow = [255 255 153]./255;        % gamma frequency DAD method (ch & epoch)

% Behavioral tags: color of shaded behavioral locations
dad.gui.cyan = [0 255 255]./255;            % sleepy/relaxed = alpha frequency DAD method (ch & epoch)
dad.gui.pink = [255 204 204]./255;          % beta frequency DAD method (ch & epoch)
dad.gui.brown = [153 76 0]./255;            % bitting/clench = gamma frequency power with z > 2, detection of 5 or fewer significant peaks

% Import test data
dad.constants.importdata_STEP1 = 1;  % Assign file path
dad.constants.importdata_STEP2 = 2;  % Import the selected data-file
dad.constants.importdata_STEP3 = 3;  % If the user made a mistake typing, try some other delimiters
dad.constants.importdata_STEP4 = 4;  % Data statistics
dad.constants.importdata_STEP5 = 5;  % Save

% Import database
dad.constants.importdatabase_STEP1 = 1; % Assign file path
dad.constants.importdatabase_STEP2 = 2; % Import the selected data-file
dad.constants.importdatabase_STEP3 = 3; % Save

% Standard method STEPS
dad.constants.stand_STEP1 = 1;  % Epoch the data into 10 second pieces
dad.constants.stand_STEP2 = 2;  % Choose a threshold for the standard method
dad.constants.stand_STEP3 = 3;  % Select cells based on threshold
dad.constants.stand_STEP4 = 4;  % Filter: Bandpass filter the data to remove frequencies greater than 40 Hz and lower than 1Hz
dad.constants.stand_STEP5 = 5;  % Short results (allREJECT, someOK)
dad.constants.stand_STEP6 = 6;  % Transform data from epochs to channels

% DAD method (whole) STEPS

% DAD method (whole) z-values


% DAD method (epochs) STEPS

% DAD method (epochs) z-values
dad.constants.DADepoch_var = 2; % (STEP e3) - Variance within epoch
dad.constants.DADepoch_hurst = 2; % (STEP e4) - Calculate the Hurst exponent within the epoch
dad.constants.DADepoch_deltafreq = 2; % (STEP e5a) - Calculate the Frequency Power (delta: 1-3) of each epoch
dad.constants.DADepoch_thetafreq = 2; %(STEP e5b) - Calculate the Frequency Power (theta: 4-7) of each epoch
dad.constants.DADepoch_alphafreq = 2.5; % (STEP e5c) - Calculate the Frequency Power (alpha: 8-15) of each epoch
dad.constants.DADepoch_betafreq = 2.5; % (STEP e5d) - Calculate the Frequency Power (beta: 16-31) of each epoch
dad.constants.DADepoch_gammafreq = 2; % (STEP e5e) - Calculate the Frequency Power (gamma: 32-49) of each epoch
dad.constants.DADepoch_peaks = 2;   % (STEP e6) - Peaks within epoch

end
% -----------------------------------------
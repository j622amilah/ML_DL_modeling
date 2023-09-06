function import_data(datatype, filename, pathname)

global dad

% ----------------------------------
screen = get(0, 'ScreenSize');
% swidth  = screen(3);
sheight = screen(4);
im = imread('eeglocs.jpg');
iwidth  = size(im,2) - 950;
iheight = size(im,1) - 950;
pos = [0 (sheight-iheight)/2 iwidth iheight];
dad.gui.eeglocs = figure('Visible', 'on', 'menubar', 'none', 'paperpositionmode', 'auto', 'numbertitle', 'off', ...
    'resize', 'off', 'position', pos, 'name', 'EEG locations');
image(im);
set(gca, 'Visible', 'off', 'Position', [0 0 1 1]);

splocation = inputdlg('Enter the spatial location(s): [ie: C3;C4]');

if exist('splocation', 'var') == 1
    delete(dad.gui.eeglocs);
end

dad.splocation = strsplit(splocation{1}, ';');

clear splocation

set(dad.popup_sploc, 'String', dad.splocation); % Set the menu to entered EEG locations
dad.popup_splocValue = get(dad.popup_sploc, 'Value'); % should be the current value
% ----------------------------------


dad.current.fileopen_ok = 0;

switch datatype
    case 'text'
        ext = {'*.txt'};
        
    otherwise
        if dad.intern.prompt
            msgbox('Unknown filetype.', 'Info')
        end
        return
end

if nargin < 3
    [filename, pathname] = uigetfile(ext, ['Choose a ', datatype, ' data-file']);
    if all(filename == 0) || all(pathname == 0) % Cancel
        return
    end
end

% ****************************
h = waitbar(0, 'Loading...');
steps = dad.constants.importdata_STEP5;     % Total number of steps to complete the computation
% ****************************


% 1. Assign file path
file = fullfile(pathname, filename);

% ****************************
curstep = dad.constants.importdata_STEP1;
waitbar(curstep/steps);
% ****************************


% 2. Import the selected data-file
try
    switch datatype
        case 'text'
            M = dlmread(sprintf('%s', file));
            
            dad.data.time = M(:, 1);    % assume the 1st column is time
            
            
            [r, c] = size(M); %#ok<ASGLU>
            
            if c > 1    % data file has at least two columns of data
                for i = 1:(c-1)
                    dad.data_eegchall{i,1} = M(:, i+1);       % Full vector of data
                end
                
            else
                errordlg('Load data such that the columns are [time, eegch1, eegch2, ...]')
            end
            
            dad.database = 0;   % dummy value for database
            
            dad.numofrawdata = c-1;
    end
catch
    sprintf('Unable to import %s', file);
    return
end

% ****************************
curstep = dad.constants.importdata_STEP2;
waitbar(curstep/steps);
% ****************************


% 3. The user made a mistake typing, try some other delimiters
if length(dad.splocation) ~= dad.numofrawdata
    dad.splocation = strsplit(dad.splocation{1}, ',');  % comma
    set(dad.popup_sploc, 'String', dad.splocation); % Set the menu to entered EEG locations
    dad.popup_splocValue = get(dad.popup_sploc, 'Value'); % should be the current value
end
if length(dad.splocation) ~= dad.numofrawdata
    dad.splocation = strsplit(dad.splocation{1});   % whitespace
    set(dad.popup_sploc, 'String', dad.splocation); % Set the menu to entered EEG locations
    dad.popup_splocValue = get(dad.popup_sploc, 'Value'); % should be the current value
end
if length(dad.splocation) ~= dad.numofrawdata
    dad.splocation = strsplit(dad.splocation{1}, '.');   % dot
    set(dad.popup_sploc, 'String', dad.splocation); % Set the menu to entered EEG locations
    dad.popup_splocValue = get(dad.popup_sploc, 'Value'); % should be the current value
end
if length(dad.splocation) ~= dad.numofrawdata
    dad.splocation = strsplit(dad.splocation{1}, '-');   % hyphen
    set(dad.popup_sploc, 'String', dad.splocation); % Set the menu to entered EEG locations
    dad.popup_splocValue = get(dad.popup_sploc, 'Value'); % should be the current value
end

% ****************************
curstep = dad.constants.importdata_STEP3;
waitbar(curstep/steps);
% ****************************


% 4. Data statistics
dad.file.filename = filename;
dad.file.pathname = pathname;
dad.file.date = clock;

dad.data.N = length(dad.data_eegchall{1,1});
dad.data.samplingrate = round((dad.data.N - 1) / (dad.data.time(end) - dad.data.time(1)));  % Fs = 256
dad.data.eegch.min = min(dad.data_eegchall{1,1});
dad.data.eegch.max = max(dad.data_eegchall{1,1});
dad.data.eegch.error = sqrt(mean(diff(dad.data_eegchall{1,1}).^2)/2);

% ****************************
curstep = dad.constants.importdata_STEP4;
waitbar(curstep/steps);
% ****************************


% 5. Save
save('dad.mat','dad');
clear c r

% ****************************
curstep = dad.constants.importdata_STEP5;
waitbar(curstep/steps);
delete(h)
% ****************************




% ----------------------------------
% Plot raw data in subfigure 1
% ----------------------------------
plot_data1;
% ----------------------------------
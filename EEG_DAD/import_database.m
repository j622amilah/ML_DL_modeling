function import_database(datatype, filename, pathname)

% If dad.mat exists, load it
if exist('dad.mat', 'file') == 2
    load dad.mat
end

global dad


switch datatype
%     case 'mat'
%         ext = {'*.mat'};
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
steps = dad.constants.importdatabase_STEP3;
% ****************************



% 1. Assign file path
file = fullfile(pathname, filename);

% ****************************
curstep = dad.constants.importdatabase_STEP1;
waitbar(curstep/steps);
% ****************************



% 2. Import the selected data-file
% [dataset1 dataset2 dataset3 ... dataset8] all from the same spatial location -> size (N x 8)
try
    switch datatype
        case 'mat',
            dad.database = load(file);
            
        case 'text'
            dad.database = dlmread(sprintf('%s', file));
    end
catch
   add2log(0, ['Unable to import ', file, '.'], 1, 1, 0, 1, 0, 1)
   return
end

% ****************************
curstep = dad.constants.importdatabase_STEP2;
waitbar(curstep/steps);
% ****************************


% 3. Save
save('dad.mat','dad');  % load dad.mat if it exist

% ****************************
curstep = dad.constants.importdatabase_STEP3;
waitbar(curstep/steps);
delete(h)
% ****************************
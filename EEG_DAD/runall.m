function runall

global dad

runStandard;

% Check if database is loaded
if dad.database == 0  % No database has been loaded
    errordlg('Load a Database (Import tab)...')
else
    runDAD_ch;
    
    runDAD_ep;
end
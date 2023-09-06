function runDAD_ch_ep

global dad

% Check if database is loaded
if dad.database == 0  % No database has been loaded
    errordlg('Load a Database (Import tab)...')
else
    runDAD_ch;
    
    runDAD_ep;
end
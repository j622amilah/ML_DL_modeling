function exit_gui

%global var1

% Save check before closing - Ask if you want to close 
%close_ledafile;

% if leda2.file.open
%     return;
% end %closing failed

delete(gcf)

%save_ledamem

% add2log(0,['<<<< ',datestr(now,31), ' Session closed'],1);
% add2log(0,' ',1);

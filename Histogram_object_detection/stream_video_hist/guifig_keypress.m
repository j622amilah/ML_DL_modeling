function guifig_keypress

global var1

ch = double(get(var1.fig_main, 'CurrentCharacter'));

if isempty(ch) %Strg / Cntrl
    return;
end
% 
% switch ch
%     case 27,  % Esc key
%         %do something
%         
%     case 110, % "n" key
%         
        
end

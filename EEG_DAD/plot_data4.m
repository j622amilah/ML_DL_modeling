function plot_data4

global dad

% Subfigure 4 - DAD method (overlay detected behavioral areas)
axes(dad.gui.overview.ax4);
cla;
hold on

dad.gui.overview.ch4 = plot(dad.data_eegch_dad10beh_full_time, dad.data_eegch_dad10beh_full, 'ButtonDownFcn', 'dad_click(4)', 'Color', dad.gui.darkblue, 'LineWidth', 1);


% Plot shaded areas over places where artifacts occur
for i = 1:length(dad.DADbadcounter_beh)
    
    switch dad.DADbadcounter_beh(i,1)
        
        case 52  % Frequency - alpha
            u1 = round((i-1)*2561);
            x1 = u1/dad.data.samplingrate;
            x2 = x1 + 10;
            dad.gui.overview.ch3 = fill([x1 x2 x2 x1], [-2 -2 2 2], dad.gui.cyan, ...
                'LineStyle', ':', 'FaceAlpha', 0.8, 'ButtonDownFcn', 'dad_click(3)');
        
        case 53  % Frequency - beta
            u1 = round((i-1)*2561);
            x1 = u1/dad.data.samplingrate;
            x2 = x1 + 10;
            dad.gui.overview.ch3 = fill([x1 x2 x2 x1], [-2 -2 2 2], dad.gui.pink, ...
                'LineStyle', ':', 'FaceAlpha', 0.8, 'ButtonDownFcn', 'dad_click(3)');
        
        case 6  % Biting/clench
            u1 = round((i-1)*2561);
            x1 = u1/dad.data.samplingrate;
            x2 = x1 + 10;
            dad.gui.overview.ch3 = fill([x1 x2 x2 x1], [-2 -2 2 2], dad.gui.brown, ...
                'LineStyle', ':', 'FaceAlpha', 0.8, 'ButtonDownFcn', 'dad_click(3)');
            
    end
    
end


clear r i x1 x2


% ------- IF YOU WANT THE Y-AXIS TO REMAIN [-2 2] -------
set(dad.gui.overview.ax4, 'XLim', [0 dad.data_eegch_dad10beh_full_time(end)], 'YLim', [-2 2], 'Color', dad.gui.lightgray)
% -----------------------------------------------------------
function plot_data3

global dad

% Subfigure 3 - DAD method (overlay detected artifact areas)
axes(dad.gui.overview.ax3);
cla;
hold on

dad.gui.overview.ch3 = plot(dad.data_eegch_dad10art_full_time, dad.data_eegch_dad10art_full, 'ButtonDownFcn', 'dad_click(3)', 'Color', dad.gui.darkblue, 'LineWidth', 1);


% Plot shaded areas over places where artifacts occur
for i = 1:length(dad.DADbadcounter)
    
    switch dad.DADbadcounter(i,1)
        
        case 3  % Variance
            u1 = round((i-1)*2561);
            x1 = u1/dad.data.samplingrate;
            x2 = x1 + 10;
            dad.gui.overview.ch3 = fill([x1 x2 x2 x1], [-2 -2 2 2], dad.gui.lightblue, ...
                'LineStyle', ':', 'FaceAlpha', 0.8, 'ButtonDownFcn', 'dad_click(3)');
        
        case 4  % Hurst Exponent
            u1 = round((i-1)*2561);
            x1 = u1/dad.data.samplingrate;
            x2 = x1 + 10;
            dad.gui.overview.ch3 = fill([x1 x2 x2 x1], [-2 -2 2 2], dad.gui.lightred, ...
                'LineStyle', ':', 'FaceAlpha', 0.8, 'ButtonDownFcn', 'dad_click(3)');
        
        case 50  % Frequency - delta
            u1 = round((i-1)*2561);
            x1 = u1/dad.data.samplingrate;
            x2 = x1 + 10;
            dad.gui.overview.ch3 = fill([x1 x2 x2 x1], [-2 -2 2 2], dad.gui.darkgray, ...
                'LineStyle', ':', 'FaceAlpha', 0.8, 'ButtonDownFcn', 'dad_click(3)');
        
        case 51  % Frequency - theta
            u1 = round((i-1)*2561);
            x1 = u1/dad.data.samplingrate;
            x2 = x1 + 10;
            dad.gui.overview.ch3 = fill([x1 x2 x2 x1], [-2 -2 2 2], dad.gui.olivegreen, ...
                'LineStyle', ':', 'FaceAlpha', 0.8, 'ButtonDownFcn', 'dad_click(3)');
        
        case 54  % Frequency - gamma
            u1 = round((i-1)*2561);
            x1 = u1/dad.data.samplingrate;
            x2 = x1 + 10;
            dad.gui.overview.ch3 = fill([x1 x2 x2 x1], [-2 -2 2 2], dad.gui.yellow, ...
                'LineStyle', ':', 'FaceAlpha', 0.8, 'ButtonDownFcn', 'dad_click(3)');
            
    end
    
end


clear r i x1 x2


% ------- IF YOU WANT THE Y-AXIS TO REMAIN [-2 2] -------
set(dad.gui.overview.ax3, 'XLim', [0 dad.data_eegch_dad10art_full_time(end)], 'YLim', [-2 2], 'Color', dad.gui.lightgray)
% -----------------------------------------------------------
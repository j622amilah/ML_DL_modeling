function edits1(flag)

global dad

switch flag
    case 1
        if isnumstr(get(dad.gui.rangeview1.edit_start, 'String'))
            dad.gui.rangeview1.start = str2double(get(dad.gui.rangeview1.edit_start, 'String'));
        end

        if isnumstr(get(dad.gui.rangeview1.edit_range, 'String'))
            dad.gui.rangeview1.range = str2double(get(dad.gui.rangeview1.edit_range, 'String'));
        end

    case 2
        rg_end = str2double(get(dad.gui.rangeview1.edit_end, 'String'));

        if rg_end > dad.gui.rangeview1.start
            dad.gui.rangeview1.range = rg_end - dad.gui.rangeview1.start;
        else
            rg_end = dad.gui.rangeview1.start + dad.gui.rangeview1.range;
            set(dad.gui.rangeview1.edit_end, 'String', rg_end);
        end

    case 3
        if exist('dad.gui.rangeview1.slider', 'var') == 1
            dad.gui.rangeview1.start = get(dad.gui.rangeview1.slider, 'Value');
        else
        end
        
    case 4
        dad.popup_splocValue = get(dad.popup_sploc, 'Value');
        plot_data1; % Replot the new 'Value' field
end



% Check if field (X-values) is within overview
if dad.gui.rangeview1.start < 0
    dad.gui.rangeview1.start = 0;
end

if dad.gui.rangeview1.range > dad.data.time(end),
    dad.gui.rangeview1.range = dad.data.time(end);
end

if (dad.gui.rangeview1.start + dad.gui.rangeview1.range) >= dad.data.time(end) 
    dad.gui.rangeview1.start = dad.data.time(end) - dad.gui.rangeview1.range;
end

set(dad.gui.rangeview1.edit_start, 'String', num2str(dad.gui.rangeview1.start, '%3.2f'))
set(dad.gui.rangeview1.edit_range, 'String', num2str(dad.gui.rangeview1.range, '%3.2f'))
set(dad.gui.rangeview1.edit_end, 'String', num2str(dad.gui.rangeview1.start + dad.gui.rangeview1.range, '%3.2f'))



% Check Y-limits for overview-field = rangeview
cond_rg.data = dad.data_eegchall{dad.popup_splocValue, 1}(subrange_idx(dad.data.time, dad.gui.rangeview1.start, dad.gui.rangeview1.start + dad.gui.rangeview1.range));
cond_rg.min = min(cond_rg.data);
cond_rg.max = max(cond_rg.data);
cond_rg.yrange = max(.5, (cond_rg.max - cond_rg.min)*1.2);
rg_bottom = (cond_rg.max + cond_rg.min)/2 - cond_rg.yrange/2;
rg_top = (cond_rg.max + cond_rg.min)/2 + cond_rg.yrange/2;
dad.gui.rangeview1.bottom = rg_bottom;
dad.gui.rangeview1.top = rg_top;
rg_start = dad.gui.rangeview1.start;
rg_end = dad.gui.rangeview1.start + dad.gui.rangeview1.range;

set(dad.gui.overview.ax1, 'XLim', [rg_start, rg_end], 'Ylim', [rg_bottom, rg_top]);


% Slider
rem = dad.data.time(end) - dad.gui.rangeview1.range;
if rem <= 0,
    rem = 2;
end

sliderstep = dad.gui.rangeview1.range/rem;
smallsliderstep = sliderstep/10;

if sliderstep > 1
    sliderstep = 1;
end

if smallsliderstep > 1
    smallsliderstep = 1;
end

set(dad.gui.rangeview1.slider, 'sliderstep', [smallsliderstep, sliderstep], 'min', 0, 'max', rem, 'Value', dad.gui.rangeview1.start)
function edits4(flag)

global dad

rgview = dad.gui.rangeview4;

switch flag
    case 1
        if isnumstr(get(rgview.edit_start, 'String'))
            rgview.start = str2double(get(rgview.edit_start, 'String'));
        end

        if isnumstr(get(rgview.edit_range, 'String'))
            rgview.range = str2double(get(rgview.edit_range, 'String'));
        end

    case 2
        rg_end = str2double(get(rgview.edit_end, 'String'));

        if rg_end > rgview.start
            rgview.range = rg_end - rgview.start;
        else
            rg_end = rgview.start + rgview.range;
            set(rgview.edit_end, 'String', rg_end);
        end

    case 3
        rgview.start = get(rgview.slider, 'Value');
end

dad.gui.rangeview4 = rgview;
rgview = dad.gui.rangeview4;
rgview.ax = dad.gui.overview.ax4;

% Check if field (X-values) is within overview
if rgview.start < 0
    rgview.start = 0;
end

if rgview.range > dad.data_eegch_dad10beh_full_time(end),
    rgview.range = dad.data_eegch_dad10beh_full_time(end);
end

if (rgview.start + rgview.range) >= dad.data_eegch_dad10beh_full_time(end) 
    rgview.start = dad.data_eegch_dad10beh_full_time(end) - rgview.range;
end

set(rgview.edit_start, 'String', num2str(rgview.start, '%3.2f'))
set(rgview.edit_range, 'String', num2str(rgview.range, '%3.2f'))
set(rgview.edit_end, 'String', num2str(rgview.start + rgview.range, '%3.2f'))

% Check Y-limits for overview-field = rangeview
cond_rg.data = dad.data_eegch_dad10beh_full(subrange_idx(dad.data_eegch_dad10beh_full_time, rgview.start, rgview.start + rgview.range));
cond_rg.min = min(cond_rg.data);
cond_rg.max = max(cond_rg.data);
cond_rg.yrange = max(0.5, (cond_rg.max - cond_rg.min)*1.2);
rg_bottom = (cond_rg.max + cond_rg.min)/2 - cond_rg.yrange/2;
rg_top = (cond_rg.max + cond_rg.min)/2 + cond_rg.yrange/2;
rgview.bottom = rg_bottom;
rgview.top = rg_top;
rg_start = rgview.start;
rg_end = rgview.start + rgview.range;

set(rgview.ax, 'XLim', [rg_start, rg_end], 'Ylim', [rg_bottom, rg_top]);

% Slider
rem = dad.data_eegch_dad10beh_full_time(end) - rgview.range;
if rem <= 0,
    rem = 2;
end

sliderstep = rgview.range/rem;
smallsliderstep = sliderstep/10;

if sliderstep > 1
    sliderstep = 1;
end

if smallsliderstep > 1
    smallsliderstep = 1;
end

set(rgview.slider, 'sliderstep', [smallsliderstep, sliderstep], 'min', 0, 'max', rem, 'Value', rgview.start)

dad.gui.rangeview4 = rgview;
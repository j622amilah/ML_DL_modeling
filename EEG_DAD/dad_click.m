function dad_click(ax_flag)

global dad

switch ax_flag
    case 1      % Subfigure 1 - Raw data
        point1 = get(dad.gui.overview.ax1, 'currentpoint');
        point2 = get(dad.gui.overview.ax1, 'currentpoint');
        point1 = point1(1, 1:2);
        point2 = point2(1, 1:2);
        pt1 = min(point1, point2); % left-bottom (x,y)
        pt2 = max(point1, point2); % right-top (x,y)
        
        if (pt1(1) > 0 - dad.data.samplingrate*10) && (pt1(1) < dad.data.N + dad.data.samplingrate*10) && (pt1(2) > 0) && (pt1(2) < dad.data.eegch.max + 1) %Hit within overview-axes
            pt1(1) = withinlimits(pt1(1), 0, dad.data.time(end));
            dad.gui.rangeview1.start = pt1(1);
            
            if norm(pt2 - pt1) > 2 && (pt2(1) > 0) && (pt2(1) < dad.data.N) && (pt2(2) > 0) && (pt2(2) < dad.data.eegch.max + 1)
                pt2(1) = withinlimits(pt2(1), 0, dad.data.time(end));
                dad.gui.rangeview1.range = pt2(1) - pt1(1);
            end
            
            % ----------------
            rgview = dad.gui.rangeview1;
            rgview.ax = dad.gui.overview.ax1;

            % Check if field (X-values) is within overview
            if rgview.start < 0
                rgview.start = 0;
            end

            if rgview.range > dad.data.time(end),
                rgview.range = dad.data.time(end);
            end

            if (rgview.start + rgview.range) >= dad.data.time(end) 
                rgview.start = dad.data.time(end) - rgview.range;
            end

            set(rgview.edit_start, 'String', num2str(rgview.start, '%3.2f'))
            set(rgview.edit_range, 'String', num2str(rgview.range, '%3.2f'))
            set(rgview.edit_end, 'String', num2str(rgview.start + rgview.range, '%3.2f'))

            % Check Y-limits for overview-field = rangeview
            dad.popup_splocValue = get(dad.popup_sploc, 'Value');
            cond_rg.data = dad.data_eegchall{dad.popup_splocValue, 1}(subrange_idx(dad.data.time, rgview.start, rgview.start + rgview.range));  %***
            cond_rg.min = min(cond_rg.data);
            cond_rg.max = max(cond_rg.data);
            cond_rg.yrange = max(.5, (cond_rg.max - cond_rg.min)*1.2);
            rg_bottom = (cond_rg.max + cond_rg.min)/2 - cond_rg.yrange/2;
            rg_top = (cond_rg.max + cond_rg.min)/2 + cond_rg.yrange/2;
            rgview.bottom = rg_bottom;
            rgview.top = rg_top;
            rg_start = rgview.start;
            rg_end = rgview.start + rgview.range;

            set(rgview.ax, 'XLim', [rg_start, rg_end], 'Ylim', [rg_bottom, rg_top]);

            % Slider
            rem = dad.data.time(end) - rgview.range;
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

            dad.gui.rangeview1 = rgview;
            % ----------------
        end
        
        set(dad.gui.rangeview1.slider, 'value', dad.gui.rangeview1.start);
        
        
        
    case 2      % Subfigure 2 - Standard method w/ artifact tags
        point1 = get(dad.gui.overview.ax2, 'currentpoint');
        point2 = get(dad.gui.overview.ax2, 'currentpoint');
        point1 = point1(1, 1:2);
        point2 = point2(1, 1:2);
        pt1 = min(point1, point2); % left-bottom (x,y)
        pt2 = max(point1, point2); % right-top (x,y)
        
        if (pt1(1) > 0 - dad.data.samplingrate*10) && (pt1(1) < dad.data.N + dad.data.samplingrate*10) && (pt1(2) > 0) && (pt1(2) < dad.data.eegch.max + 1) %Hit within overview-axes
            pt1(1) = withinlimits(pt1(1), 0, dad.data.time(end));
            dad.gui.rangeview2.start = pt1(1);
            
            if norm(pt2 - pt1) > 2 && (pt2(1) > 0) && (pt2(1) < dad.data.N) && (pt2(2) > 0) && (pt2(2) < dad.data.eegch.max + 1)
                pt2(1) = withinlimits(pt2(1), 0, dad.data.time(end));
                dad.gui.rangeview2.range = pt2(1) - pt1(1);
            end
            
            % ----------------
            rgview = dad.gui.rangeview2;
            rgview.ax = dad.gui.overview.ax2;

            % Check if field (X-values) is within overview
            if rgview.start < 0
                rgview.start = 0;
            end

            if rgview.range > dad.data.time(end),
                rgview.range = dad.data.time(end);
            end

            if (rgview.start + rgview.range) >= dad.data.time(end) 
                rgview.start = dad.data.time(end) - rgview.range;
            end

            set(rgview.edit_start, 'String', num2str(rgview.start, '%3.2f'))
            set(rgview.edit_range, 'String', num2str(rgview.range, '%3.2f'))
            set(rgview.edit_end, 'String', num2str(rgview.start + rgview.range, '%3.2f'))

            % Check Y-limits for overview-field = rangeview
            cond_rg.data = dad.data_eegch_stand10_r2full(subrange_idx(dad.data.time, rgview.start, rgview.start + rgview.range));    % ****
            cond_rg.min = min(cond_rg.data);

            cond_rg.max = max(cond_rg.data);
            cond_rg.yrange = max(.5, (cond_rg.max - cond_rg.min)*1.2);
            rg_bottom = (cond_rg.max + cond_rg.min)/2 - cond_rg.yrange/2;
            rg_top = (cond_rg.max + cond_rg.min)/2 + cond_rg.yrange/2;
            rgview.bottom = rg_bottom;
            rgview.top = rg_top;
            rg_start = rgview.start;
            rg_end = rgview.start + rgview.range;

            set(rgview.ax, 'XLim', [rg_start, rg_end], 'Ylim', [rg_bottom, rg_top]);

            % Slider
            rem = dad.data.time(end) - rgview.range;
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

            dad.gui.rangeview2 = rgview;
            % ----------------
        end
        
        set(dad.gui.rangeview2.slider, 'value', dad.gui.rangeview2.start);
    
    
    
    
    case 3      % Subfigure 3 - DAD method w/ artifact tags
        point1 = get(dad.gui.overview.ax3, 'currentpoint');
        point2 = get(dad.gui.overview.ax3, 'currentpoint');
        point1 = point1(1, 1:2);
        point2 = point2(1, 1:2);
        pt1 = min(point1, point2); % left-bottom (x,y)
        pt2 = max(point1, point2); % right-top (x,y)
        
        if (pt1(1) > 0 - dad.data.samplingrate*10) && (pt1(1) < dad.data.N + dad.data.samplingrate*10) && (pt1(2) > 0) && (pt1(2) < dad.data.eegch.max + 1) %Hit within overview-axes
            pt1(1) = withinlimits(pt1(1), 0, dad.data.time(end));
            dad.gui.rangeview3.start = pt1(1);
            
            if norm(pt2 - pt1) > 2 && (pt2(1) > 0) && (pt2(1) < dad.data.N) && (pt2(2) > 0) && (pt2(2) < dad.data.eegch.max + 1)
                pt2(1) = withinlimits(pt2(1), 0, dad.data.time(end));
                dad.gui.rangeview3.range = pt2(1) - pt1(1);
            end
            
            % ----------------
            rgview = dad.gui.rangeview3;
            rgview.ax = dad.gui.overview.ax3;

            % Check if field (X-values) is within overview
            if rgview.start < 0
                rgview.start = 0;
            end

            if rgview.range > dad.data.time(end),
                rgview.range = dad.data.time(end);
            end

            if (rgview.start + rgview.range) >= dad.data.time(end) 
                rgview.start = dad.data.time(end) - rgview.range;
            end

            set(rgview.edit_start, 'String', num2str(rgview.start, '%3.2f'))
            set(rgview.edit_range, 'String', num2str(rgview.range, '%3.2f'))
            set(rgview.edit_end, 'String', num2str(rgview.start + rgview.range, '%3.2f'))

            % Check Y-limits for overview-field = rangeview
            cond_rg.data = dad.data_eegch_dad10art_full(subrange_idx(dad.data.time, rgview.start, rgview.start + rgview.range));    % ****
            cond_rg.min = min(cond_rg.data);

            cond_rg.max = max(cond_rg.data);
            cond_rg.yrange = max(.5, (cond_rg.max - cond_rg.min)*1.2);
            rg_bottom = (cond_rg.max + cond_rg.min)/2 - cond_rg.yrange/2;
            rg_top = (cond_rg.max + cond_rg.min)/2 + cond_rg.yrange/2;
            rgview.bottom = rg_bottom;
            rgview.top = rg_top;
            rg_start = rgview.start;
            rg_end = rgview.start + rgview.range;

            set(rgview.ax, 'XLim', [rg_start, rg_end], 'Ylim', [rg_bottom, rg_top]);

            % Slider
            rem = dad.data.time(end) - rgview.range;
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

            dad.gui.rangeview3 = rgview;
            % ----------------
        end
        
        set(dad.gui.rangeview3.slider, 'value', dad.gui.rangeview3.start);
    
    
    
    
    case 4      % Subfigure 4 - DAD method w/ behavior tags
        point1 = get(dad.gui.overview.ax4, 'currentpoint');
        point2 = get(dad.gui.overview.ax4, 'currentpoint');
        point1 = point1(1, 1:2);
        point2 = point2(1, 1:2);
        pt1 = min(point1, point2); % left-bottom (x,y)
        pt2 = max(point1, point2); % right-top (x,y)
        
        if (pt1(1) > 0 - dad.data.samplingrate*10) && (pt1(1) < dad.data.N + dad.data.samplingrate*10) && (pt1(2) > 0) && (pt1(2) < dad.data.eegch.max + 1) %Hit within overview-axes
            pt1(1) = withinlimits(pt1(1), 0, dad.data.time(end));
            dad.gui.rangeview4.start = pt1(1);
            
            if norm(pt2 - pt1) > 2 && (pt2(1) > 0) && (pt2(1) < dad.data.N) && (pt2(2) > 0) && (pt2(2) < dad.data.eegch.max + 1)
                pt2(1) = withinlimits(pt2(1), 0, dad.data.time(end));
                dad.gui.rangeview4.range = pt2(1) - pt1(1);
            end
            
            % ----------------
            rgview = dad.gui.rangeview4;
            rgview.ax = dad.gui.overview.ax4;

            % Check if field (X-values) is within overview
            if rgview.start < 0
                rgview.start = 0;
            end

            if rgview.range > dad.data.time(end),
                rgview.range = dad.data.time(end);
            end

            if (rgview.start + rgview.range) >= dad.data.time(end) 
                rgview.start = dad.data.time(end) - rgview.range;
            end

            set(rgview.edit_start, 'String', num2str(rgview.start, '%3.2f'))
            set(rgview.edit_range, 'String', num2str(rgview.range, '%3.2f'))
            set(rgview.edit_end, 'String', num2str(rgview.start + rgview.range, '%3.2f'))

            % Check Y-limits for overview-field = rangeview
            cond_rg.data = dad.data_eegch_dad10beh_full(subrange_idx(dad.data.time, rgview.start, rgview.start + rgview.range));    % ****
            cond_rg.min = min(cond_rg.data);

            cond_rg.max = max(cond_rg.data);
            cond_rg.yrange = max(.5, (cond_rg.max - cond_rg.min)*1.2);
            rg_bottom = (cond_rg.max + cond_rg.min)/2 - cond_rg.yrange/2;
            rg_top = (cond_rg.max + cond_rg.min)/2 + cond_rg.yrange/2;
            rgview.bottom = rg_bottom;
            rgview.top = rg_top;
            rg_start = rgview.start;
            rg_end = rgview.start + rgview.range;

            set(rgview.ax, 'XLim', [rg_start, rg_end], 'Ylim', [rg_bottom, rg_top]);

            % Slider
            rem = dad.data.time(end) - rgview.range;
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
            % ----------------
        end
        
        set(dad.gui.rangeview4.slider, 'value', dad.gui.rangeview4.start);
        
    
    
    
end


% -----------------------------------------
function w_out = withinlimits(w_in, lowerlimit, upperlimit)

w_out = max(min(w_in, upperlimit), lowerlimit);
% -----------------------------------------
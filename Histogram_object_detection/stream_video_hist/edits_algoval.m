function var1 = edits_algoval(flag, var1)

% rgview = leda2.gui.rangeview;

switch flag
    case 1
        var1.algoOpt1 = str2double(get(var1.det_ax_algoOpt1,'String'));
        
%     case 2,
%         rg_end = str2double(get(rgview.edit_end,'String'));
%         if rg_end > rgview.start
%             rgview.range = rg_end - rgview.start;
%         else
%             rg_end = rgview.start + rgview.range;
%             set(rgview.edit_end,'String',rg_end);
%         end
%         
%     case 3, 
%         rgview.start = get(rgview.slider,'Value');
%         
%     case 4,
%         
%         leda2.gui.overview.min = str2double(get(leda2.gui.overview.edit_min,'String'));
%         set(leda2.gui.overview.ax, 'Ylim',[leda2.gui.overview.min, leda2.gui.overview.max]);
%         refresh_fitoverview;
%         
%     case 5,
%         eventnr = str2double(get(leda2.gui.eventinfo.edit_eventnr,'String'));
%                 eventnr = withinlimits(eventnr, 1, leda2.data.events.N);
% 
%     case 6,
%         if leda2.gui.eventinfo.current_event
%             eventnr = leda2.gui.eventinfo.current_event - 1;
%             eventnr = max(1, eventnr);
%         else
%             eventnr = find([leda2.data.events.event.time] < leda2.gui.rangeview.start);
%             if ~isempty(eventnr)
%                 eventnr = eventnr(end);
%             else
%                 eventnr = 1;
%             end
%         end
%         
%     case 7,
%         if leda2.gui.eventinfo.current_event
%             eventnr = leda2.gui.eventinfo.current_event + 1;
%             eventnr = min(leda2.data.events.N, eventnr);
%         else
%             eventnr = find([leda2.data.events.event.time] > leda2.gui.rangeview.start);
%             if ~isempty(eventnr)
%                 eventnr = eventnr(1);
%             else
%                 eventnr = leda2.data.events.N;
%             end
%         end
end


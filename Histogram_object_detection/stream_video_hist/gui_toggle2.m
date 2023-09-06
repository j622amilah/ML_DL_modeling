function gui_toggle2(var1)

fig_main = figure('Units','normalized','Name', [var1.name,' ',num2str(var1.version)],'KeyPressFcn','guifig_keypress', 'Position',[0.1 0.1 0.8 0.8], ...
    'MenuBar','none','NumberTitle','off','Color', var1.gui_fig_color, 'CloseRequestFcn','delete(gcf)');

var1.overview_ax = axes('Units','normalized','Position',[.2 .5 .5 .4]); %,'ButtonDownFcn','leda_click(1)'     [50 70 60 40]
set(var1.overview_ax,'XLim',[0, var1.w_reduce],'YLim',[0, var1.h_reduce], 'Color',[0.9 0.9 0.9]);

var1.det_ax = axes('Units','normalized','Position',[.73 .5 .2 .2]); %,'ButtonDownFcn','leda_click(1)'     [50 70 60 40]
set(var1.det_ax,'XLim',[0, var1.w_reduce],'YLim',[0, var1.h_reduce], 'Color',[0.9 0.9 0.9]);

var1.det_ax_algoOpt1 = uicontrol('Units','normalized','Style','edit','Position',[.95 .5 .04 .025], 'String','0.85');     %,'Callback','edits_algoval(1)');
var1.det_ax_algoOpt2 = uicontrol('Units','normalized','Style','edit','Position',[.95 .55 .04 .025], 'String','1');

var1.axis_searchobj = axes('Units','normalized','Position',[.02 .6 .1 .1]); %,'ButtonDownFcn','leda_click(1)'     [50 70 60 40]
set(var1.axis_searchobj,'XLim',[0, var1.w_reduce],'YLim',[0, var1.h_reduce], 'Color',[0.9 0.9 0.9]);

var1.axis_xmove = axes('Units','normalized','Position',[.1 .2 .8 .1]); %,'ButtonDownFcn','leda_click(1)'     [50 70 60 40]
set(var1.axis_xmove, 'XLim', [1, var1.num_of_frames], 'YLim',[var1.centroid_xmove_min-1, var1.centroid_xmove_max+1], 'Color', [0.9 0.9 0.9]);

var1.axis_ymove = axes('Units','normalized','Position',[.1 .05 .8 .1]); %,'ButtonDownFcn','leda_click(1)'     [50 70 60 40]
set(var1.axis_ymove, 'XLim', [1, var1.num_of_frames], 'YLim', [var1.centroid_ymove_min-1, var1.centroid_ymove_max+1], 'Color', [0.9 0.9 0.9]);

var1.instruction = uicontrol('style','text', 'units','pix', 'position',[500 780 400 40], 'string', '       Video Stream       ', 'fontsize', 20);
set(var1.instruction,'backgroundcolor', get(fig_main, 'color'))

conbut = uicontrol('Style','toggle', 'Units','pix', 'Position',[30 650 70 50], 'CallBack', {@callb, var1}, 'String','Camera off');

end



function var1 = callb(conbut, event, var1)
    
    set(conbut, 'string', 'Camera on')
    drawnow
    
    var1.num_of_frames = 1;
    
    videoObject = cam(1, var1);     % START video stream
    
    while 1     % a trigger statement here will never stop the while loop, just put something that is always true
        
        % ------------------------------
        %tstart = tic;
        
        [videoObject, vid2_mat, var1] = cam(2, var1, videoObject);     % GET a FRAME from the video stream
        vid2_pix = double(vid2_mat)/255;      % size h , w
        clear vid2_mat

        %tcam_load = toc(tstart)
        % ------------------------------
        
        % ------------------------------
        %tstart = tic;
        
        % 3b. Reduces the video size
        U_reduce = zeros(var1.h_reduce, var1.w_reduce, var1.num_of_frames_per_grab);

        pixel(:,:,var1.num_of_frames_per_grab) = (rgb2gray(vid2_pix(:,:,:, var1.num_of_frames_per_grab)));     % only takes one frame at a time
        
        U_reduce(:,:,var1.num_of_frames_per_grab) = imresize(pixel(:,:,var1.num_of_frames_per_grab), [var1.h_reduce, var1.w_reduce]);

        clear  pixel vid2_pix
        
        % Display each video frame
        axes(var1.overview_ax);
        cla;
        hold on

        % Check to see if an ROI was taken, if not display the reduced video
        if exist('roicomplete','var') == 0
            var1.overview_ax_reducedvideo = imshow(U_reduce(:, :, var1.num_of_frames_per_grab), []);
        end

        %tvideo_plot = toc(tstart)
        % ------------------------------
        
        
        % ------------------------------
        % Re-initialize edit entries per loop
        var1.algoOpt1 = str2double(get(var1.det_ax_algoOpt1,'String'));
        var1.run_falsedetection_removal = str2double(get(var1.det_ax_algoOpt2,'String'));
        % ------------------------------
        
        
        if var1.num_of_frames  == 1
            
            % 같같같같같같같같같같같같같같같
            % Get region of interest on first loop
            % 같같같같같같같같같같같같같같같
            set(var1.instruction, 'String', 'Select region of interest:')
            rect = getrect;
            set(var1.instruction, 'String', '       Video Stream       ')

            x_min = round(rect(1,2));
            y_min = round(rect(1,1));
            if x_min > 1
                var1.x_vec_w = x_min:(round(rect(1,4))+x_min-1);
            else
                var1.x_vec_w = x_min:round(rect(1,4));
            end

            if y_min > 1
                var1.y_vec_h = y_min:(round(rect(1,3))+y_min-1);
            else
                var1.y_vec_h = y_min:round(rect(1,3));
            end
            
            % Save all frames in Uroi
            Uroi(:,:,var1.num_of_frames) = U_reduce(var1.x_vec_w, var1.y_vec_h, var1.num_of_frames_per_grab);     % h, w is reversed: above order is h, w.  here it is w, h
            roicomplete = 1;        % ROI was created

            clear x_min y_min U_reduce

            var1.overview_ax_reducedvideo = imshow(zeros(var1.w, var1.h, 1), []);  % fill-in background black
            var1.overview_ax_reducedvideo = imshow(Uroi(:,:,var1.num_of_frames), []);
            
            var1.rows = size(Uroi, 1);      % h_roi
            var1.cols = size(Uroi, 2);      % w_roi

            % 같같같같같같같같같같같같같같같
            % Find a unique feature to find in the image
            % 같같같같같같같같같같같같같같같
            set(var1.instruction, 'String', ' Select a unique feature: ')
            rect2 = getrect;
            set(var1.instruction, 'String', '       Video Stream       ')

            x_min = round(rect2(1,2));
            y_min = round(rect2(1,1));
            if x_min > 1
                var1.x_vec2_w = x_min:(round(rect2(1,4))+x_min-1);
            else
                var1.x_vec2_w = x_min:round(rect2(1,4));
            end

            if y_min > 1
                var1.y_vec2_h = y_min:(round(rect2(1,3))+y_min-1);
            else
                var1.y_vec2_h = y_min:round(rect2(1,3));
            end
            
            org = Uroi(var1.x_vec2_w, var1.y_vec2_h, var1.num_of_frames);     % h, w is reversed: above order is h, w.  here it is w, h

            axes(var1.axis_searchobj)   % select search object axes
            cla;
            var1.overview_searchobj = imshow(org(:,:,var1.num_of_frames), []);     % assign area grab to the axes of search object

            clear x_min y_min

            var1.a = size(org, 1);
            var1.aw = size(org, 2);
        else
            
            % Save all frames in Uroi
            Uroi(:,:,var1.num_of_frames) = U_reduce(var1.x_vec_w, var1.y_vec_h, var1.num_of_frames_per_grab);     % h, w is reversed: above order is h, w.  here it is w, h

            clear U_reduce
            var1.overview_ax_reducedvideo = imshow(Uroi(:,:,var1.num_of_frames), []);
        end
        
        
        var1 = idea2_tracker2(Uroi, org, rect, rect2, var1);
        axes(var1.axis_xmove)   % select x direction movement axes
        plot(1:var1.num_of_frames, var1.x_center, 'b')

        axes(var1.axis_ymove)   % select y direction movement axes
        plot(1:var1.num_of_frames, var1.y_center, 'b')
        
        
        % Note: Need to dump Uroi in a .mat file every 100 frames to prevent slowing down
        if rem(var1.num_of_frames, 100) == 0
            timestamp = datestr(now,'yyyy-mm-dd HH:MM:SS.FFF');
            dt_cell = textscan(timestamp, '%s', 'Delimiter', {'-', ':', ' ', '.'});
            out = sprintf('y%sm%sd%s_h%s_m%s_s%s_f%s', char(dt_cell{1}(1)), char(dt_cell{1}(2)), char(dt_cell{1}(3)), char(dt_cell{1}(4)), char(dt_cell{1}(5)), char(dt_cell{1}(6)), char(dt_cell{1}(7)));
            
            save(sprintf('%s.mat', out), 'Uroi', '-v7.3');
            clear Uroi timestamp dt_cell out
        end
        
        
        
        
        
        var1.num_of_frames = var1.num_of_frames + 1;
        drawnow
        
        if ~get(conbut, 'value')    % a trigger statement with an "if statement" is the only thing that will stop the while loop (Camera On = 1, Camera Off = 0) 
            
            var1 = cam(3, var1, videoObject);       % STOP video stream - as long as it does not crash before you can push the button, the videoObject will be deleted, so you can restart the gui
           
            set(conbut, 'string', 'Camera off')
            
            timestamp = datestr(now,'yyyy-mm-dd HH:MM:SS.FFF');
            dt_cell = textscan(timestamp, '%s', 'Delimiter', {'-', ':', ' ', '.'});
            out = sprintf('y%sm%sd%s_h%s_m%s_s%s_f%s', char(dt_cell{1}(1)), char(dt_cell{1}(2)), char(dt_cell{1}(3)), char(dt_cell{1}(4)), char(dt_cell{1}(5)), char(dt_cell{1}(6)), char(dt_cell{1}(7)));
            
            save(sprintf('var1_%s.mat', out), 'var1', '-v7.3');
            save(sprintf('Uroi_%s.mat', out), 'Uroi', '-v7.3');
            break
        end
    end
    
    
end
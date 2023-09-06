function guuii

global var1

% 'Position',[0 0 1 1]   .00 .03 1 .92
var1.fig_main = figure('Units','normalized','Name', [var1.name,' ',num2str(var1.version)],'KeyPressFcn','guifig_keypress', 'Position',[0.1 0.1 0.8 0.8], ...
    'MenuBar','none','NumberTitle','off','Color', var1.gui_fig_color, 'CloseRequestFcn','delete(gcf)');  % 'CloseRequestFcn','exit_gui', 'outerposition',[

%leda2.gui.overview.edit_max = uicontrol('Units','normalized','Style','edit','Position',[.94 dy+.155 .04 .025],'String','20','Callback','edits_cb(4)');

%var1.start_but = uicontrol('Style','pushbutton', 'String','Start tracking', 'Callback', 'start_str', 'Position',[50 20 60 40]);
%var1.stop_but = uicontrol('Style','pushbutton', 'String','Stop tracking', 'Callback', 'stopit', '', 'Position',[50 70 60 40]);

var1.conbut = uicontrol('Style','toggle', 'Units','pix', 'Position',[30 650 70 50], 'CallBack', @callb, 'String','Camera off');

var1.overview_ax = axes('Units','normalized','Position',[.2 .5 .5 .4]); %,'ButtonDownFcn','leda_click(1)'     [50 70 60 40]
set(var1.overview_ax,'XLim',[0, var1.w_reduce],'YLim',[0, var1.h_reduce], 'Color',[0.9 0.9 0.9]);  % 'title', 'Video stream'

var1.axis_searchobj = axes('Units','normalized','Position',[.02 .6 .1 .1]); %,'ButtonDownFcn','leda_click(1)'     [50 70 60 40]
set(var1.axis_searchobj,'XLim',[0, var1.w_reduce],'YLim',[0, var1.h_reduce], 'Color',[0.9 0.9 0.9]);  % 'title', 'Video stream'

var1.axis_xmove = axes('Units','normalized','Position',[.1 .2 .8 .1]); %,'ButtonDownFcn','leda_click(1)'     [50 70 60 40]
set(var1.axis_xmove, 'XLim', [1, var1.num_of_frames], 'YLim',[var1.centroid_xmove_min-1, var1.centroid_xmove_max+1], 'Color', [0.9 0.9 0.9]);

var1.axis_ymove = axes('Units','normalized','Position',[.1 .05 .8 .1]); %,'ButtonDownFcn','leda_click(1)'     [50 70 60 40]
set(var1.axis_ymove, 'XLim', [1, var1.num_of_frames], 'YLim', [var1.centroid_ymove_min-1, var1.centroid_ymove_max+1], 'Color', [0.9 0.9 0.9]);

var1.instruction = uicontrol('style','text', 'units','pix', 'position',[500 780 400 40], 'string', '       Video Stream       ', 'fontsize', 20);
set(var1.instruction,'backgroundcolor', get(var1.fig_main, 'color'))

end



function [] = callb(St, event)

global var1

set(St, 'string', 'Camera on')
drawnow

var1.num_of_frames = 1;

cam(1);     % START video stream

while 1     % a trigger statement here will never stop the while loop, just put something that is always true
    
    % ------------------------------
    tstart = tic;
    
    cam(2);     % GET a FRAME from the video stream
    var1.vid2_pix = double(var1.vid2_mat)/255;      % size h , w
    var1 = rmfield(var1, 'vid2_mat');
    
    tcam_load = toc(tstart)
    % ------------------------------
    
    
    % ------------------------------
    tstart = tic;
    % 3b. Reduces the video size
    U_reduce = zeros(var1.h_reduce, var1.w_reduce, var1.num_of_frames_per_grab);
    
    pixel(:,:,var1.num_of_frames_per_grab) = (rgb2gray(var1.vid2_pix(:,:,:, var1.num_of_frames_per_grab)));     % only takes one frame at a time
    
    
    U_reduce(:,:,var1.num_of_frames_per_grab) = imresize(pixel(:,:,var1.num_of_frames_per_grab), [var1.h_reduce, var1.w_reduce]);
    
    clear  pixel
    var1 = rmfield(var1, 'vid2_pix');

    % Display each video frame
    axes(var1.overview_ax);
    cla;
    hold on
    
    % Check to see if an ROI was taken, if not display the reduced video
    if exist('roicomplete','var') == 0
        var1.overview_ax_reducedvideo = imshow(U_reduce(:, :, var1.num_of_frames_per_grab), []);
    end
    
    tvideo_plot = toc(tstart)
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
        var1.Uroi(:,:,var1.num_of_frames) = U_reduce(var1.x_vec_w, var1.y_vec_h, var1.num_of_frames_per_grab);     % h, w is reversed: above order is h, w.  here it is w, h
        roicomplete = 1;        % ROI was created
        
        clear rect x_min y_min U_reduce
        
        var1.overview_ax_reducedvideo = imshow(zeros(var1.w, var1.h, 1), []);  % fill-in background black
        var1.overview_ax_reducedvideo = imshow(var1.Uroi(:,:,var1.num_of_frames), []);
        
        
        var1.rows = size(var1.Uroi, 1);
        var1.cols = size(var1.Uroi, 2);
        
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

        var1.org = var1.Uroi(var1.x_vec2_w, var1.y_vec2_h, var1.num_of_frames);     % h, w is reversed: above order is h, w.  here it is w, h
        
        axes(var1.axis_searchobj)   % select search object axes
        cla;
        var1.overview_searchobj = imshow(var1.org(:,:,var1.num_of_frames), []);     % assign area grab to the axes of search object

        clear rect2 x_min y_min
        
        var1.a = size(var1.org, 1);
        var1.aw = size(var1.org, 2);
        
        %[x_center, y_center, area_found] = idea2_tracker(dstep_row, dstep_acr, h_roi, w_roi, a, aw, Uroi, org, num_of_frames);
        
    else
        % Save all frames in Uroi
        var1.Uroi(:,:,var1.num_of_frames) = U_reduce(var1.x_vec_w, var1.y_vec_h, var1.num_of_frames_per_grab);     % h, w is reversed: above order is h, w.  here it is w, h
        
        clear U_reduce
        var1.overview_ax_reducedvideo = imshow(var1.Uroi(:,:,var1.num_of_frames), []);
        
        %[x_center, y_center, area_found] = idea2_tracker(dstep_row, dstep_acr, h_roi, w_roi, a, aw, Uroi, org, num_of_frames, x_center, y_center, area_found);
    end
    
    get(St, 'value')
  
    var1.num_of_frames = var1.num_of_frames + 1
    
    if ~get(St, 'value')    % a trigger statement with an "if statement" is the only thing that will stop the while loop
        %cam(3);       % STOP video stream - as long as it does not crash before you can push the button, the videoObject will be deleted, so you can restart the gui
        
        get(St, 'value')
        stop(var1.videoObject);
        delete(var1.videoObject);
        
        set(St, 'string', 'Camera off')
        break
    end
end

end
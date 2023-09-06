
%%

info1 = mmfileinfo('C:\Users\Jamilah\Desktop\Motion_detection\TRecord\combined_JA_d08m03y17.avi');

%%

aviinfo('C:\Users\Jamilah\Desktop\Motion_detection\TRecord\combined_JA_d08m03y17.avi')

%%

clear all
close all
clc

videoname = 'out';
%load(sprintf('var_%s.mat', videoname), 'svar');

paths.main_path = 'C:\Users\Jamilah\Desktop\Motion_detection\read_video_hist';
cd(paths.main_path);
%paths.testing = 'C:\Users\Jamilah\Desktop\Motion_detection\TRecord\infraliminary_d27m02y17\';
paths.testing = 'C:\Users\Jamilah\Desktop\Motion_detection\TRecord\';

%http://icephoenix.us/notes-for-myself/auto-splitting-video-file-in-equal-chunks-with-ffmpeg-and-python/

% Initialize
svar.run_falsedetection_removal = 0;

vid = VideoReader([paths.testing, videoname, '.mp4']);                               % get duration (seconds)
%vid = VideoReader([paths.testing, videoname, '.avi']);
svar.total_frames = vid.NumberOfFrames;                % OR size(vid2.pix,4) if you read in the entire matrix via read(vid) - but this is impossible for large videos


%%

for current_frame_num = 279:svar.total_frames           %1:svar.total_frames	 % Number of 5 second segments
    % ----------------------------------------
    % 1. Read the video information: # frames per second, RGB24 640x352, total number of frames in video available
    % ----------------------------------------
    vid = VideoReader([paths.testing, videoname, '.mp4']);
    %vid = VideoReader([paths.testing, videoname, '.avi']);
    
    w = vid.Width;                                          % get width
    h = vid.Height;                                         % get height
    duration = vid.Duration;                                % get duration (seconds)
    svar.total_frames = vid.NumberOfFrames;                % OR size(vid2.pix,4) if you read in the entire matrix via read(vid) - but this is impossible for large videos
    svar.fr = vid.FrameRate;
    %fprintf('Reading video %s, with %d frame, currently at frame %d\r', videoname, svar.total_frames, current_frame_num)
    current_frame_num
    
    % ----------------------------------------
    % 2. Read one frame at a time
    % ----------------------------------------
    vid2.mat = read(vid, current_frame_num);
    
    % ----------------------------------------
    % 3. Preprocessing
    % ----------------------------------------
    
    % 3a. Normalize image
    vid2.pix = double(vid2.mat)/255;
    
    % 3b. Reduces the video size
    h_small = round(h/3);
    w_small = round(w/3);
    %U = zeros(h_small, w_small);
    
    pixel = (rgb2gray(vid2.pix));
    U = imresize(pixel, [h_small, w_small]);
    
    clear vid pixel h_small w_small
    
    
    
    % ----------------------------------------
    % 4. Take ROI and feature from the first segment only, and reuse for all the segments
    % ----------------------------------------
    if current_frame_num == 1
        [svar, smats] = get_region_interest(U, svar, current_frame_num);   % calls asign_U_to_Uroi, U is cleared

        svar = get_feature_obj(svar, smats);    % org is in svar - it is reused for all segments

        svar.dstep_row = 2;         % ************** Change - can make smaller to make the comparison more accurate
        svar.dstep_acr = 2;
    else
        smats = asign_U_to_Uroi(U, svar, current_frame_num);       % U is cleared
    end
    
    
    % ----------------------------------------
    % 5. Histogram tracker program
    % ----------------------------------------
    [svar, smats, x_center, y_center, area_found] = idea2_tracker2(svar, smats, current_frame_num);
    
    
    
    % ----------------------------------------
    % 6. Save tracker result to text file every frame
    % ----------------------------------------
    if current_frame_num == 1
        svar.y_center_first = y_center;
    end
    y_dir_move = (svar.y_center_first - y_center) + svar.y_center_first;%(smats.y_center(1,1) - smats.y_center(ind,1)) + smats.y_center(1,1);  % inverse the direction so it is more natural/easy to understand
    
    q_measure = abs(smats.area_org_scale - area_found);%abs(smats.area_org_scale - smats.area_found(ind,1));
    
    if exist(sprintf('%s2.txt', videoname),'file') == 0 %if file does not exist
        fs = fopen(sprintf('%s2.txt', videoname), 'w'); % make file
    else
        fs = fopen(sprintf('%s2.txt', videoname),  'a+');   % append to file
    end
    fprintf(fs, '%d      %f      %f      %f\r\n', current_frame_num, x_center, y_dir_move, q_measure);
    fclose(fs);
    
%     smats.x_center(current_frame_num, 1) = x_center;
%     smats.y_center(current_frame_num, 1) = y_center;
%     smats.area_found(current_frame_num, 1) = area_found;
    
    % ----------------------------------------
    % 6. Save video matrix and tracker result in smat, save reused variable information in svar
    % ----------------------------------------
%     if rem(current_frame_num, 50) == 0
%         save(sprintf('mat_%d.mat', current_frame_num), 'smats', '-v7.3');     % save every video, x_center, y_center, area_found
%         
%         % Keep all data: both false (0) and good detection 
%         %ind = find(x_center ~= 0); % to only keep good detection
%         x_dir_move = smats.x_center;
%         
%         
% 
% 
% 
%         for z = 1:length(x_dir_move)
%             if exist(sprintf('%s2.txt', videoname),'file') == 0 %if file does not exist
%                 fileID = fopen(sprintf('%s2.txt', videoname), 'w'); % make file
%             else
%                 fileID = fopen(sprintf('%s2.txt', videoname),  'a+');   % append to file
%             end
%             fprintf(fileID, '%d      %f      %f      %f\r\n', z, x_dir_move(z), y_dir_move(z), q_measure(z));
%             fclose(fileID);
%         end
%         
%         % Free up memory
%         clear smats x_dir_move y_dir_move q_measure
%     end
    
    
end     % end of current_frame_num

save(sprintf('var_%s.mat', videoname), 'svar', '-v7.3');      % org, total_frames, .. other important variables to evaluate other related video segments


%%

clear all
close all
clc

videoname = 'N170227_161148_161648';
load(sprintf('var_%s.mat', videoname), 'svar');

paths.main_path = 'C:\Users\Jamilah\Desktop\Motion_detection\read_video_hist';
cd(paths.main_path);
paths.testing = [paths.main_path, '\TRecord\infraliminary_d27m02y17\JA_d28m02y17_segments\'];


%http://icephoenix.us/notes-for-myself/auto-splitting-video-file-in-equal-chunks-with-ffmpeg-and-python/


for num_of_segs = 9:24	 % Number of 5 second segments
    
    svar.num_of_segs = num_of_segs; 
    svar.run_falsedetection_removal = 1;
    
    % ----------------------------------------
    % 1. Load the data 
    % ----------------------------------------
    subname = sprintf('%s_%d', videoname, num_of_segs);
    fprintf('Video %s\r', subname)
    
    
    % ----------------------------------------
    % 2. Read one video at a time
    % ----------------------------------------
    vid = VideoReader(['C:\Users\Jamilah\Desktop\Motion_detection\TRecord\infraliminary_d27m02y17\JA_d28m02y17_segments\', subname, '.mp4']);          % create VideoReader object
    w = vid.Width;                                                % get width
    h = vid.Height;                                               % get height
    duration = vid.Duration;                                      % get duration (seconds)
    svar.num_of_frames = vid.NumberOfFrames;     % OR size(vid2.pix,4);
    svar.fr = vid.FrameRate;
    fprintf('Number of frames: %d\r', svar.num_of_frames)
    vid2.mat = read(vid, inf);
    
    % ----------------------------------------
    % 3. Preprocessing
    % ----------------------------------------
    
    % 3a. Normalize image
    vid2.pix = double(vid2.mat)/255;
    
    % 3b. Reduces the video size
    h_small = round(h/3);
    w_small = round(w/3);
    U = zeros(h_small, w_small, svar.num_of_frames);
    
    for f = 1:svar.num_of_frames
        pixel(:,:,f) = (rgb2gray(vid2.pix(:,:,:,f)));
        U(:,:,f) = imresize(pixel(:,:,f), [h_small, w_small]);
    end
    
    clear f vid pixel h_small w_small
    
    
    
    % ----------------------------------------
    % 4. Take ROI and feature from the first segment only, and reuse for all the segments
    % ----------------------------------------
    if num_of_segs == 1
        [svar, smats] = get_region_interest(U, svar);   % calls asign_U_to_Uroi, U is cleared

        svar = get_feature_obj(svar, smats);    % org is in svar - it is reused for all segments

        svar.dstep_row = 2;         % ************** Change - can make smaller to make the comparison more accurate
        svar.dstep_acr = 2;
    else
        smats = asign_U_to_Uroi(U, svar);       % U is cleared
    end
    
    
    % ----------------------------------------
    % 5. Histogram tracker program
    % ----------------------------------------
    for f = 1:svar.num_of_frames
        [svar, smats] = idea2_tracker2(svar, smats, f);
    end
    
    
    % ----------------------------------------
    % 6. Save video matrix and tracker result in smat, save reused variable information in svar
    % ----------------------------------------
    save(sprintf('mat_%d.mat', num_of_segs), 'smats', '-v7.3');     % save every video, x_center, y_center, area_found 
    save(sprintf('var_%s.mat', videoname), 'svar', '-v7.3');      % org, num_of_frames, .. other important variables to evaluate other related video segments

    
    % Keep all data: both false (0) and good detection 
    %ind = find(x_center ~= 0); % to only keep good detection
    x_dir_move = smats.x_center;
    y_dir_move = (smats.y_center(1,1) - smats.y_center) + smats.y_center(1,1);%(smats.y_center(1,1) - smats.y_center(ind,1)) + smats.y_center(1,1);  % inverse the direction so it is more natural/easy to understand
    q_measure = abs(smats.area_org_scale - smats.area_found);%abs(smats.area_org_scale - smats.area_found(ind,1));
    
    
    
    for z = 1:svar.num_of_frames
        if exist(sprintf('%s2.txt', videoname),'file') == 0 %if file does not exist
            fileID = fopen(sprintf('%s2.txt', videoname), 'w'); % make file
        else
            fileID = fopen(sprintf('%s2.txt', videoname),  'a+');   % append to file
        end
        fprintf(fileID, '%d      %f      %f      %f\r\n', z, x_dir_move(z), y_dir_move(z), q_measure(z));
        fclose(fileID);
    end
    
    
    % Free up memory
    clear smats x_dir_move y_dir_move q_measure
    
    %load(sprintf('var_%s.mat', videoname), 'svar');  % org, num_of_frames, .. other important variables to evaluate other related video segments
    
end     % end of num_of_segs



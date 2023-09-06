%%

num_of_segs = 1;
videoname = 'N170227_161148_161648';
    subname = sprintf('%s_%d', videoname, num_of_segs);
    fprintf('Video %s\r', subname)
    
    
    % ----------------------------------------
    % 2. Read one frame at a time
    % ----------------------------------------
    vid = VideoReader(['C:\Users\Jamilah\Desktop\Motion_detection\TRecord\infraliminary_d27m02y17\', videoname, '.mp4']);
    
    out = read(vid, 1);
    
    %%
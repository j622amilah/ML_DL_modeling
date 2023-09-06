

%%
% imaqtool - Call the Image Acquisition Toolbox

% To reinstall the camera software/hardware connections: run ../imaq/installgenicam.m

% AdaptorName: 'winvideo'
% DeviceName: 'HP HD Webcam'
% MaxHeight: 480
% MaxWidth: 640
% NativeDataType: 'uint8'
% TotalSources: 1
% VendorDriverDescription: 'Windows WDM Compatible Driver'
% VendorDriverVersion: 'DirectX 9.0'

close all
clear all
clc

tstart = tic;
% General: set up video input object
adaptorName = 'winvideo';
deviceID = 1;
vidFormat = 'MJPG_640x360';    % 'RGB24_640x480';

videoObject = videoinput(adaptorName, deviceID, vidFormat);
w = 640;                                             
h = 360;

set(videoObject, 'FramesPerTrigger', 5);        % videoObject.FramesPerTrigger = 2;
set(videoObject, 'FrameGrabInterval', 1);       % amount of time to record video data

% Region of Interest
%videoObject.ROIPosition = [0 0 640 480];  % Location of video and region of interest

% Device Properties (brightness, white balance, saturation, hue, sharpness)
% src.FrameRate = '20.0000';
%src.Exposure = -2;

%videoObject.ReturnedColorSpace = 'grayscale';
%videoObject.DeviceProperties.Brightness = 150;

imaqhwinfo(videoObject)     % Displays information about the video stream

%preview(videoObject)                          % view the video streaming, Does not record
% close(adaptorName, deviceID, vidFormat)       % close the view of video streaming

%frame = step(videoObject);      % Acquire a single frame  - does not work with Matlab 2012
%frame = imaqmontage(videoObject);       %Display a sequence of image frames as a montage

start(videoObject);

    vid2.mat = getdata(videoObject);   % get frame 
toc(tstart)


tstart = tic;
    % ----------------------------------------
    % 3. Preprocessing
    % ----------------------------------------
    % 3a. Normalize image
    vid2.pix = double(vid2.mat)/255;
    
    % 3b. Reduces the video size
    nFrames = size(vid2.pix, 4);
    h_reduce = round(h/3);
    w_reduce = round(w/3);
    U_reduce = zeros(h_reduce, w_reduce, 1);        %zeros(h_reduce, w_reduce, num_of_frames) - only keep one frame
    
for num_of_frames = 1:nFrames
    pixel(:,:,num_of_frames) = (rgb2gray(vid2.pix(:,:,:,num_of_frames))); %#ok<SAGROW>
    U_reduce(:,:,num_of_frames) = imresize(pixel(:,:,num_of_frames), [h_reduce, w_reduce]);    % U_reduce(:,:,num_of_frames)

figure
        imshow(U_reduce(:, :, num_of_frames), []);      % imshow(U_reduce(:, :, num_of_frames), []);
        pause(0.05);
        close all
end

stop(videoObject);

toc(tstart)

%%

%videoObject.ReturnedColorspace = 'grayscale';   % change color space to grayscale
keep_running = 1;
num_of_frames = 1; % count the number of times the while loop runs


while keep_running == 1
    
    start(videoObject);
    stoppreview(videoObject);
    vid2.mat = getdata(videoObject);   % get frame 
    
    % ----------------------------------------
    % 3. Preprocessing
    % ----------------------------------------
    % 3a. Normalize image
    vid2.pix = double(vid2.mat)/255;
    
    % 3b. Reduces the video size
    nFrames = size(vid2.pix,4);
    h_reduce = round(h/3);
    w_reduce = round(w/3);
    U_reduce = zeros(h_reduce, w_reduce, 1);        %zeros(h_reduce, w_reduce, num_of_frames) - only keep one frame
    
    pixel(:,:,num_of_frames) = (rgb2gray(vid2.pix(:,:,:,num_of_frames))); %#ok<SAGROW>
    U_reduce(:,:,1) = imresize(pixel(:,:,num_of_frames), [h_reduce, w_reduce]);    % U_reduce(:,:,num_of_frames)
    clear  vid2 pixel
    
    % 같같같같같같같같같같같같같같같
    % Get region of interest on first loop
    if num_of_frames == 1
        % 3c. Region of interest
        figure
        imshow(U_reduce(:, :, 1), []);      % imshow(U_reduce(:, :, num_of_frames), []);
        title('Select the region in which you are interested in:')
        rect = getrect;
        close all
        
        x_min = round(rect(1,2));
        y_min = round(rect(1,1));
        if x_min > 1
            x_vec = x_min:(round(rect(1,4))+x_min-1);
        else
            x_vec = x_min:round(rect(1,4));
        end
        
        if y_min > 1
            y_vec = y_min:(round(rect(1,3))+y_min-1);
        else
            y_vec = y_min:round(rect(1,3));
        end
        
        h_roi = rect(1, 4);
        w_roi = rect(1, 3);
    end
    
    clear rect x_min y_min
    % 같같같같같같같같같같같같같같같
    
    Uroi(:,:,num_of_frames) = U_reduce(x_vec, y_vec, 1);    %#ok<SAGROW> % Uroi(:,:,num_of_frames) = U_reduce(x_vec, y_vec, num_of_frames);
    
    clear U_reduce
    
    figure
    title('Video playback of Region of interest')
    imshow(Uroi(:, :, num_of_frames), []);
    axis image off
    drawnow;
    
    
    rows = size(Uroi, 1);
    cols = size(Uroi, 2);
    
    % 같같같같같같같같같같같같같같같
    % find a unique feature to find in the image
    if num_of_frames == 1
        
        figure
        imshow(Uroi(:, :, 1), []);     % take from the first frame only
        title('Select a search object:')
        rect2 = getrect;
        close all
        
        x_min = round(rect2(1,2));
        y_min = round(rect2(1,1));
        if x_min > 1
            x_vec2 = x_min:(round(rect2(1,4))+x_min-1);
        else
            x_vec2 = x_min:round(rect2(1,4));
        end

        if y_min > 1
            y_vec2 = y_min:(round(rect2(1,3))+y_min-1);
        else
            y_vec2 = y_min:round(rect2(1,3));
        end

        org = Uroi(x_vec2, y_vec2, 1);         %  ? org = U(x_vec2, y_vec2, 2);

        clear rect2 x_min y_min
        
        figure
        imshow(org)
        title('Search object')
    end
    % 같같같같같같같같같같같같같같같
    
    a = size(org, 1); 
    aw = size(org, 2);

    dstep_row = 2;         % ************** Change - can make smaller to make the comparison more accurate
    dstep_acr = 2;

    
    if num_of_frames == 1
        [x_center, y_center, area_found] = idea2_tracker(dstep_row, dstep_acr, h_roi, w_roi, a, ...
            aw, Uroi, org, num_of_frames);
    else
        [x_center, y_center, area_found] = idea2_tracker(dstep_row, dstep_acr, h_roi, w_roi, a, ...
            aw, Uroi, org, num_of_frames, x_center, y_center, area_found);
    end
    
    
    num_of_frames = num_of_frames + 1;

end     % end of while loop keep_running



%%



%%
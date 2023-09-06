function varargout = cam(flagger, var1, varagin)

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


% Collect a time stamp everytime the camera is accessed
var1.time_samp(var1.num_of_frames, 1) = now;        % [first value=start cam, getsnapshot, last value=stop cam], ==> to get string of date and time (timestamp = datestr(now,'yyyy-mm-dd HH:MM:SS.FFF');)


switch flagger
    case 1      % START video stream
        % General: set up video input object
        
        % Logitech Webcam C210 (winvideo-1)
        adaptorName = 'winvideo';
        deviceID = 1;
        vidFormat = 'RGB24_640x480';
        
        % Internal HP HD Webcam (winvideo-2)
%         adaptorName = 'winvideo';
%         deviceID = 2;
%         vidFormat = 'MJPG_640x360';
        
        videoObject = videoinput(adaptorName, deviceID, vidFormat);

        set(videoObject, 'FramesPerTrigger', var1.num_of_frames_per_grab);        % videoObject.FramesPerTrigger = 2;
        set(videoObject, 'FrameGrabInterval', 1);
        set(videoObject, 'TriggerFrameDelay', 0);       % videoObject.TriggerFrameDelay = 0;

        %set(var1.videoObject, 'ReturnedColorSpace', 'rgb')

        imaqhwinfo(videoObject)     % Displays information about the video stream
        % preview(videoObject)        % view the video streaming, Does not record

        triggerconfig(videoObject, 'manual');  % Put the video trigger into 'manual', this starts streaming the video
        % without saving it. We can then request frames at will while only having
        % to run the startup overhead this one time.

        start(videoObject);
        
        varargout{1} = videoObject;
        varargout{2} = var1;
        
    case 2      % GET a FRAME from the video stream
        
        videoObject = varagin;
        
        %var1.vid2_mat = getdata(var1.videoObject);   % get frame 
        vid2_mat = getsnapshot(videoObject);   % get frame - faster than 
        
        varargout{1} = videoObject;
        varargout{2} = vid2_mat;
        varargout{3} = var1;
        
    case 3      % STOP video stream
        
        videoObject = varagin;
        
        stop(videoObject);
        delete(videoObject);   % delete the videoObject so it does not occupy the device ID, can re-run the gui as many times without error "The device associated with device ID 1 is already in use"
        
        varargout{1} = var1;
end
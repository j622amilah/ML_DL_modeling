function gui_name

clc
close all
clear all

var1.name = 'gui_name';
var1.version = 1;

main_path = 'C:\Users\Jamilah\Desktop\Motion_detection\stream_video_hist';

% Add subdirectories 
%addpath(genpath('c:/matlab/myfiles')) 

% ---------------------------------
% Load initial settings
% ---------------------------------
var1.gui_fig_color = [122 147 162]./255;

var1.num_of_frames_per_grab = 1;

var1.w = 640;   % width of original input video                                       
var1.h = 360;   % height of original input video

var1.w_reduce = round(var1.w/3);     % width of REDUCED input video
var1.h_reduce = round(var1.h/3);     % height of REDUCED input video

var1.dstep_row = 2;     % Change: step size in the y direction, to compare the search object with equivalent area on the ROI image  - can make smaller to make the comparison more accurate
var1.dstep_acr = 2;     % Change: step size in the x direction, to compare the search object with equivalent area on the ROI image  - can make smaller to make the comparison more accurate

var1.num_of_frames = 100;
var1.centroid_xmove_min = 0;
var1.centroid_xmove_max = var1.w_reduce;
var1.centroid_ymove_min = 0;
var1.centroid_ymove_max = var1.h_reduce;


% ---------------------------------
% Load/show opening window
% ---------------------------------
var1 = guilogo(main_path, var1);
pause(1);
delete(var1.fig_logo);

% ---------------------------------
% Launch gui
% ---------------------------------
gui_toggle2(var1);
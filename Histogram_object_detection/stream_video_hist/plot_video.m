function [h_reduce, w_reduce] = plot_video(h, w, num_of_frames)

global var1 

% ----------------------------------------
% 3. Preprocessing
% ----------------------------------------
% 3a. Normalize image
var1.vid2.pix = double(var1.vid2.mat)/255;

% 3b. Reduces the video size
h_reduce = round(h/3);
w_reduce = round(w/3);
var1.U_reduce = zeros(h_reduce, w_reduce, 1);        %zeros(h_reduce, w_reduce, num_of_frames) - only keep one frame

pixel(:,:,num_of_frames) = (rgb2gray(var1.vid2.pix(:,:,:,num_of_frames)));
var1.U_reduce(:,:,1) = imresize(pixel(:,:,num_of_frames), [h_reduce, w_reduce]);    % U_reduce(:,:,num_of_frames)
clear  vid2 pixel

%OVERVIEW (Data Display)
axes(var1.overview_ax);
cla;
hold on
var1.overview_ax_reducedvideo = imshow(var1.U_reduce(:, :, 1), []); %plot(time.data, cond.data, 'Color',[0 0 0],'LineWidth',1); %'ButtonDownFcn','leda_click(1)',

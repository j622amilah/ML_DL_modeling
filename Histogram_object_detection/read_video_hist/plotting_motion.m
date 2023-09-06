function plotting_motion

%%
close all
clear all
clc

% ------------------------------------------------------
% Read the saved text columns (x, y) from both python and matlab scripts

% PYTHON
mainpath = 'C:\Users\Jamilah\Desktop\vshare\';
%file = 'log6_JA_d08m03y17_dv';
file = 'log6_JA_d08m03y17_dv';

% MATLAB
% mainpath = 'C:\Users\Jamilah\Desktop\Motion_detection\read_video_hist\';
% file = 'N170227_161148_1616482';
%file = 'out';
% ------------------------------------------------------


save_path = mainpath;
filepath = [mainpath, file, '.txt'];

scriptdata = menu('Select type of data:', 'python', 'matlab');



A = importdata(sprintf('%s', filepath));

% get time 
t(1,1) = A(1,1);
for i = 2:size(A, 1)
    t(i,1) = A(i,1) + t(i-1,1);
end
    

switch scriptdata
    case 1
        x_dir_move = -(A(:,2)-A(1,2)) + A(1,2);     %video is flipped vertically, so need to flip the signal across the horizontal axis
        y_dir_move = A(:,4);
        q_measure = A(:,8);
        datatype = 'python';
        
        % Remove outliers: for python algo poor quality is indicated by a large value greater than 100 
        for i = 1:length(q_measure)
           if (q_measure(i,1) < 20) || (q_measure(i,1) > 60)
               if i == 1
                   x_dir_move_remOutliers(i,1) = 0;
                   y_dir_move_remOutliers(i,1) = 0;
               else
                   %x_dir_move_remOutliers(i,1) = 0;
                   %y_dir_move_remOutliers(i,1) = 0;
                   x_dir_move_remOutliers(i,1) = x_dir_move_remOutliers(i-1,1);
                   y_dir_move_remOutliers(i,1) = y_dir_move_remOutliers(i-1,1);
               end
           else
               x_dir_move_remOutliers(i,1) = x_dir_move(i,1);
               y_dir_move_remOutliers(i,1) = y_dir_move(i,1);
           end
        end
        
    case 2
        x_dir_move = A(:,2);
        y_dir_move = A(:,3);
        q_measure = A(:,4);
        datatype = 'matlab';
        
        % Remove outliers: for matlab algo poor quality is indicated by a large value greater than 100 
        for i = 1:length(q_measure)
           if q_measure(i,1) > 80%100
               if i == 1
                   x_dir_move_remOutliers(i,1) = 0;
                   y_dir_move_remOutliers(i,1) = 0;
               else
                   %x_dir_move_remOutliers(i,1) = 0;
                   %y_dir_move_remOutliers(i,1) = 0;
                   x_dir_move_remOutliers(i,1) = x_dir_move_remOutliers(i-1,1);
                   y_dir_move_remOutliers(i,1) = y_dir_move_remOutliers(i-1,1);
               end
           else
               x_dir_move_remOutliers(i,1) = x_dir_move(i,1);
               y_dir_move_remOutliers(i,1) = y_dir_move(i,1);
           end
        end
        
        load(['C:\Users\Jamilah\Desktop\Motion_detection\read_video_hist\var_', file, '.mat'])
end


num_of_frames = length(x_dir_move);
duration = t(length(t),1);

% % Save plotted movement data
% 
% Ts = duration/num_of_frames;    
% fs = fr;    % roughly 15 - 100 frames per ~6.5 secs
% t = ind.*Ts;    % sec
% 
% 
% 
% x_out = fft(x_dir_move);
% L = length(x_out);
% P2 = abs(x_out/L);
% P1 = P2(1:L/2+1);
% %P1(2:end-1) = 2*P1(2:end-1);
% f = fs*(0:(L/2))/L;
% % figure
% % plot(f,P1)
% 
% fc = 7;
% [b,a] = butter(6, fc/(fs/2));
% x_dir_move_filt = filter(b, a, x_dir_move);
% y_dir_move_filt = filter(b, a, y_dir_move);





% x,y against frames
figure('Visible', 'on')
subplot(3,1,1)
plot(x_dir_move, 'b')
hold on
%plot(x_dir_move_filt, '--b')
plot(1:length(x_dir_move), x_dir_move(1,1).*ones(1, length(x_dir_move)), '--k')
plot(x_dir_move_remOutliers, 'r')
%plot(1:length(ind), centroid_org_scale_x.*ones(1, length(ind)), '--g')  % to account for the offset, can align centroid of area found with centroid of orignal 
ylabel('x dir: forward (below), back (above)')
if strcmp(datatype, 'matlab') == 1
    ylim([0 svar.w_small])
end

subplot(3,1,2)
%plot(y_dir_move(ind,1))  % up (below line) and down (above line)
plot(y_dir_move, 'b')
hold on
%plot(y_dir_move_filt, '--b')
plot(1:length(y_dir_move), y_dir_move(1,1).*ones(1, length(y_dir_move)), '--k')
plot(y_dir_move_remOutliers, 'r')
%plot(1:length(ind), centroid_org_scale_y.*ones(1, length(ind)), '--g')  % to account for the offset, can align centroid of area found with centroid of orignal
ylabel('y dir: down (below), up (above)')
if strcmp(datatype, 'matlab') == 1
    ylim([0 svar.h_small])
end

subplot(3,1,3)
plot(q_measure, 'b')
ylabel('Accuracy')  % the greater the area difference the more erroneous the found region,  difference in area of svar.org and found

xlabel('Number of frames', 'Fontsize', 12)

set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1])
set(gcf,'PaperPositionMode','auto')
set(gcf,'PaperPosition', [1 1 28 19]);    %change the paper position instead of the position
print(gcf, '-dpng', sprintf('%sfig1_xandyVSframes_%s_%s', save_path, file, datatype)) % '-depsc'

%%






% x,y against time
% figure('Visible', 'on')
% subplot(2,1,1)
% plot(t, x_dir_move(ind,1))
% hold on
% plot(t, x_dir_move(1,1).*ones(1, length(t)), '--r')
% plot(t, centroid_org_scale_x.*ones(1, length(t)), '--g')
% ylabel('x dir: forward (below), back (above)')
% 
% subplot(2,1,2)
% %plot(t, y_dir_move(ind,1))  % up (below line) and down (above line)
% 
% plot(t, out)
% hold on
% plot(t, y_dir_move(1,1).*ones(1, length(t)), '--r')
% plot(t, centroid_org_scale_y.*ones(1, length(t)), '--g')
% ylabel('y dir: down (below), up (above)')
% xlabel('Time (sec)', 'Fontsize', 12)
% 
% set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1])
% set(gcf,'PaperPositionMode','auto')
% set(gcf,'PaperPosition', [1 1 28 19]);    %change the paper position instead of the position
% print(gcf, '-dpng', sprintf('%sfig2_xandyVStime_%s', save_path, subname)) % '-depsc'




%%

clear all
close all
clc

paths.main_path = 'C:\Users\Jamilah\Desktop\Motion_detection\';
cd(paths.main_path);
paths.testing = [paths.main_path, '\TRecord\infraliminary_d27m02y17\'];


% ----------------------------------------
% 1. Load the data - read video files in folder if they exist, else read video files in the folders
% ----------------------------------------
% fold_file = dir(paths.testing);
% 
% num_of_folders = length(fold_file);
% 
% c = 1;
% for i = 3:num_of_folders
%     
%     if exist([paths.testing, sprintf('%s', fold_file(i).name)], 'dir') == 7
%         % read each of the files in the folders only
%     else
%         % files exist in directory, read these in comparison to the folders
%         datset{c,1} = fold_file(i).name;
%         c = c + 1;
%     end
%     
% end
% 
% clear c num_of_folders


% 1 (PN) and 3 (GV). T2C experiment - screen is not fully bright, so no reflection on headset/cabin
% 2. Flying in city - JA - bright screen, reflection on headset/cabin (can do greyscale, subtract, etc)
%
% 17/01/2017
% 3 = Video short_PN_2016_10_05.mp4
% 1 = Video short_GV_2016_09_22.mp4
% 2 = Video short_JA_2016_10_06.mp4



for i = 1:1%1:1% 2:2 %    %1:length(datset)
    
    %subname = datset{i,1}(7:8);
    %fprintf('Video %s\r', datset{i,1})
    
    %subname = 'N170227_161148_161648_1to5sec';
    subname = 'N170227_161148_161648_10to20sec';
    fprintf('Video %s\r', subname)
    
    
    % ----------------------------------------
    % 2. Read one video at a time
    % ----------------------------------------
    %vid = VideoReader([paths.testing, datset{i,1}]);              % create VideoReader object
    vid = VideoReader([paths.testing, subname, '.mp4']);              % create VideoReader object
    w = vid.Width;                                                % get width
    h = vid.Height;                                               % get height
    duration = vid.Duration;                                      % get duration (seconds)
    num_of_frames = vid.NumberOfFrames;
    fr = vid.FrameRate;
    fprintf('Number of frames: %d\r', num_of_frames)
    vid2.mat = read(vid);
    
    % ----------------------------------------
    % 3. Preprocessing
    % ----------------------------------------
    
    % 3a. Normalize image
    vid2.pix = double(vid2.mat)/255;
    
    % 3b. Reduces the video size
    nFrames = size(vid2.pix,4);
    h_small = round(h/3);
    w_small = round(w/3);
    U = zeros(h_small, w_small, num_of_frames);
    
    for f = 1:nFrames
        pixel(:,:,f) = (rgb2gray(vid2.pix(:,:,:,f)));
        U(:,:,f) = imresize(pixel(:,:,f), [h_small, w_small]);
    end
    
    % 3c. Region of interest
    
    % ++++++++++++++++++++ Plotting ++++++++++++++++++++
    figure
    imshow(U(:, :, 1), []);
    title('Select the region in which you are interested in:')
    rect = getrect;
    close all
    % +++++++++++++++++++++++++++++++++++++++++++++++++++
    
    for f = 1:nFrames
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
        
        Uroi(:,:,f) = U(x_vec, y_vec, f);
        
        % Can filter the image in a way such that the rbg information
        % is optimal for grayscale->black_and_white  - need to look at the color histogram
        % luminocity, hue, intensity, saturation step - Need to determine which is better
        
        % High pass filter
        % Filter 2
        % kernel2 = [-1 -2 -1; -2 12 -2; -1 -2 -1]/16;
        % filteredImage = imfilter(single(Uroi1(:,:,f)), kernel2); % single is needed so it can be floating point which allows the image to have negative values
        % filteredImage = imfilter(single(U(:,:,f)), kernel2);
        
        % Filter 1
        % kernel1 = -1 * ones(3)/9;
        % kernel1(2,2) = 8/9;
        % filteredImage = imfilter(single(Uroi(:,:,f)), kernel1);
        
        % Tune contrast via histogram of image
        % Uroi2(:,:,f) = histeq(Uroi(:,:,f));
        
        % figure
        % subplot(1,3,1)
        % imshow(Uroi1(:,:,f))
        % title('Original')
        
        % subplot(1,3,2)
        % imshow(Uroi2(:,:,f))
        % imshow(filteredImage, []);
        % title('Contrasted')
        
        % subplot(1,3,3)
        % imshow(filteredImage)
        % title('Filtered')
        
        % subplot(2,2,4)
        % imhist(Uroi(:,:,f))
    end
    
    % ++++++++++++++++++++ Plotting ++++++++++++++++++++
    figure
    title('Video playback of Region of interest')
    for k = 1:num_of_frames
        imshow(Uroi(:, :, k), []);
        axis image off
        drawnow;
    end
    % +++++++++++++++++++++++++++++++++++++++++++++++++++
    
    %idea2_tracker(Uroi, duration, num_of_frames, rect, subname, nFrames);
    
end     % end of subjects



%

% ----------------------------------------
% Idea 2: find a unique feature to find in the image
% ----------------------------------------
U = Uroi;

rows = size(U, 1);
cols = size(U, 2);

% Can select a region of the original picture that you want to use as a
% marker - can track this selected region in the image
% Q: when is it best to do this?  
% 1. color with original image, 
% 2. grayscale w/ original image, 
% 3. grayscale of difference of frames 
% 4. black and white of difference of frames

% grayscale w/ original image

% ++++++++++++++++++++ Plotting ++++++++++++++++++++
figure
imshow(U(:, :, 1), []);
title('Select a search object:')
rect2 = getrect;
close all
% +++++++++++++++++++++++++++++++++++++++++++++++++++

% figure
% Unew = zeros(rows,cols);
% coua = 1;
% 
% for a = 1:2:rows
%     coub = 1;
%     for b = 1:2:cols
%         Unew(a,b) = U(a,b,1);
%         Unew2(coua,coub) = U(a,b,1);
%         coub = coub + 1;
%     end
%     coua = coua + 1;
% end
% imagesc(Unew)
% 
% figure
% imagesc(Unew2(1:41, 1:42))

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

org = U(x_vec2, y_vec2, 1);     % ? org = U(x_vec2, y_vec2, 2);

% ++++++++++++++++++++ Plotting ++++++++++++++++++++
% figure
% imshow(org)
% title('Search object')
% +++++++++++++++++++++++++++++++++++++++++++++++++++

% % black and white
% hp0 = bw(1:round(h_small/3), 126:157, 2);
% hp1 = bw(1:round(h_small/3), 126:157, 5);
% hp2 = bw(1:round(h_small/3), 126:157, 12);
% 
% figure
% subplot(2,2,1)
% imshow(org)
% drawnow;
% axis image on
% 
% subplot(2,2,2)
% imshow(hp0)
% drawnow;
% axis image on
% 
% subplot(2,2,3)
% imshow(hp1)
% drawnow;
% axis image on
% 
% subplot(2,2,4)
% imshow(hp2)
% drawnow;
% axis image on

a = size(org, 1); 
aw = size(org, 2);

dstep_row = 2;         % ************** Change - can make smaller to make the comparison more accurate
dstep_acr = 2;

% qh = 1;
% qw = 1;

h_small = rect(1, 4);
w_small = rect(1, 3);

%

for f = 1:nFrames
    b = 0;
    prs = 1;
    c_row = 1;
    
    
    for picrows = 1:dstep_row:(h_small-a)
        prs = picrows;
        prf = a + picrows-1;
        c_across = 1;
        % slide feature across original image like convolution - checking for a similarity using
        % translation - not good for rotations of the same feature (need a different way to quantify a feature in space)
        
        for picacross = 1:dstep_acr:(w_small-aw)
            ccc = aw + picacross-1;
            dmatch{c_row, c_across} = abs(org - U(prs:prf, picacross:ccc, f));
            
            fee = imhist(org);          % *************************
            sect = imhist(U(prs:prf, picacross:ccc, f)); % *************************
            
            histout{c_row, c_across} = abs(fee - sect);
            histoutmean(c_row, c_across) = mean(histout{c_row, c_across});
            
            qq(c_row, c_across) = mean(mean(dmatch{c_row, c_across}));
            
            %[mssim(c_row, c_across) ssim_map] = ssim_index(org, U(prs:prf, picacross:ccc, f));  % similarity
            
            c_across = c_across + 1;
        end
        
        b = b + 1;
        c_row = c_row + 1;
    end

    
    % ---------------------------------------------------
    % Check to see if current frame is similar to first frame - skip (or do another type of analysis to track) frames where lighting dramatically changes
    if f == 1
        histoutmean_1store = histoutmean;
    end
    histoutmean_diff = abs(histoutmean_1store - histoutmean);
    %fprintf('frame %d, mean diff %f\r', f, mean(mean(histoutmean_diff)))
    
    if mean(mean(histoutmean_diff)) < 0.13
        % Perform tracking
        
        % ++++++++++++++++++++ Plotting ++++++++++++++++++++
%         if rem(f, 20)
%             figure
%             imagesc(histoutmean)
%             colorbar
%         end
        % +++++++++++++++++++++++++++++++++++++++++++++++++++
        
        [r1, r2] = size(histoutmean);

        % detect pixels that are strongly similar to the goal/feature image
        [y,i] = sort(min(histoutmean));
        num = floor(length(y)*0.1);         % weird error with floor 25/01/17 - replace function if it happens again %num = length(y)*0.1
        bin_threshhold = y(1,num);
        [ro, co] = find(histoutmean < bin_threshhold);
        histoutmean_bin = zeros(r1,r2);
        for i = 1:length(ro)
            histoutmean_bin(ro(i),co(i)) = 1;
        end
        
        
        % ++++++++++++++++++++ Plotting ++++++++++++++++++++
%         figure
%         imagesc(histoutmean_bin)
%         colorbar
        % +++++++++++++++++++++++++++++++++++++++++++++++++++
        
        % Remove false detection of pixel not belonging to the main/largest group of detected pixels
        [y, ind] = sort(co); %#ok<ASGLU>
        for j = 1:length(ind)
            co1(j,1) = co(ind(j,1),1);
            ro1(j,1) = ro(ind(j,1),1);
        end
        clear y ind co

        % determine the number of groups of detected pixels 
        grop_int = 1;
        grop(1,1) = 1;
        cc = 1;
        rr_grop_before = ro1(1,1);

        for i = 1:length(ro1)-1
           if (co1(i+1,1) == co1(i,1)+1) || (co1(i+1,1) == co1(i,1)-1) || (co1(i+1,1) == co1(i,1))
               mean_rr_grop = mean([rr_grop_before; ro1(i+1,1)]);
               rr_grop_before = [rr_grop_before; ro1(i+1,1)];

               if abs(ro1(i+1,1)-mean_rr_grop) < 4
                   grop(i+1,1) = grop_int;
                   cc = cc + 1;
               else
                   rr_grop_before = ro1(i+1,1);

                   grop_int = grop_int + 1;
                   grop(i+1,1) = grop_int;
                   cc = 1;
               end
           else
               rr_grop_before = ro1(i+1,1);

               grop_int = grop_int + 1;
               grop(i+1,1) = grop_int;
               cc = 1;
           end
        end
        
        [vals, loc, ci] = unique(grop);
        
        grop_s(1,1) = loc(1);
        for i = 2:length(vals)
            grop_s(i,1) = loc(i)-loc(i-1);
        end

        % only keep the largest group of detected pixels
        [y, b_grop_loc] = max(grop_s);
        clear grop_s % y grop_s vals ci grop

        % get new co1 and ro1 
        if b_grop_loc == 1
            co1_bgrop = co1(1:loc(b_grop_loc));
            ro1_bgrop = ro1(1:loc(b_grop_loc));
        else
            co1_bgrop = co1((loc(b_grop_loc-1)+1):loc(b_grop_loc));
            ro1_bgrop = ro1((loc(b_grop_loc-1)+1):loc(b_grop_loc));
        end

        %clear b_grop_loc co1 ro1

        % assign detected pixels from largest group
        for i = 1:length(ro1_bgrop)
            histoutmean_bin(ro1_bgrop(i),co1_bgrop(i)) = 1;
        end
        
        % Ensure that the size of the largest group is equivalent to the size of the orginal goal/feature image
        sr = regionprops(histoutmean_bin, 'BoundingBox');
        
        s = regionprops(histoutmean_bin, 'centroid');
        centroids = cat(1, s.Centroid);
        % **************************
        % Get measure of error/reliability by comparing with original position and size of "feature" image
        %scale_xmin = ((length(1:dstep_acr:(w_small-a)))*rect2(1,1))/rect(1, 3);
        %scale_ymin = ((length(1:dstep_row:(h_small-aw)))*rect2(1,2))/rect(1, 4);
        scale_xmin = ((length(1:dstep_acr:cols))*rect2(1,1))/rect(1, 3);
        scale_ymin = ((length(1:dstep_row:rows))*rect2(1,2))/rect(1, 4);
        
        scalefactor_x = round((w_small-aw-1)/rect2(1, 3));
        scalefactor_y = round((h_small-a-1)/rect2(1, 4));
        scalefactor_x2 = ((length(1:dstep_acr:(w_small-a)))*rect2(1,3))/rect(1, 3);     % original size
        scalefactor_y2 = ((length(1:dstep_row:(h_small-aw)))*rect2(1,4))/rect(1, 4);
        
        area_org_scale = scalefactor_x*scalefactor_y;
        area_found(f,1) = sr(1).BoundingBox(1,3)*sr(1).BoundingBox(1,4);
        centroid_org_scale_x = scale_xmin+(scalefactor_x/2);
        centroid_org_scale_y = scale_ymin+(scalefactor_y/2);
        
        x_center(f,1) = centroids(:,1);
        y_center(f,1) = centroids(:,2);
        % **************************
        
        % ++++++++++++++++++++ Plotting ++++++++++++++++++++
%         if rem(f, 20)
%             figure
%             imagesc(histoutmean_bin)
%             colorbar
%             hold on
%             plot(centroids(:,1), centroids(:,2), 'r*')
%             plot(centroid_org_scale_x, centroid_org_scale_y, 'g*')
%         
%             rectangle('Position', sr(1).BoundingBox, 'EdgeColor', 'r');
%             rectangle('Position', [scale_xmin, scale_ymin, scalefactor_x, scalefactor_y], 'EdgeColor', 'g');
%             rectangle('Position', [scale_xmin, scale_ymin, scalefactor_x2, scalefactor_y2], 'EdgeColor', 'm');
%         end
%         
%         if rem(f,20) == 0
%             close all 
%         end
        % +++++++++++++++++++++++++++++++++++++++++++++++++++
        
        % ---------------------------------------------------
        
    else
        x_center(f,1) = 0;
        y_center(f,1) = 0;
        
        area_found(f,1) = 100;      % big number to represent that there is no value
        
    end     % end of if mean(mean(histoutmean_diff)) < 0.13
        
        
end     % end of f

close all


ind = find(x_center ~= 0);
Ts = duration/num_of_frames;    
fs = fr;    % roughly 15 - 100 frames per ~6.5 secs
t = ind.*Ts;    % sec

save_path = 'C:\Users\Jamilah\Desktop\Motion_detection\read_video_hist\';

x_dir_move = x_center(ind,1);
y_dir_move = (y_center(1,1) - y_center(ind,1)) + y_center(1,1);  % inverse the direction so it is more natural/easy to understand

x_out = fft(x_dir_move);
L = length(x_out);
P2 = abs(x_out/L);
P1 = P2(1:L/2+1);
%P1(2:end-1) = 2*P1(2:end-1);
f = fs*(0:(L/2))/L;
% figure
% plot(f,P1)

fc = 7;
[b,a] = butter(6, fc/(fs/2));
x_dir_move_filt = filter(b, a, x_dir_move);
y_dir_move_filt = filter(b, a, y_dir_move);

figure('Visible', 'on')
subplot(2,1,1)
plot(x_dir_move, 'b')
hold on
%plot(x_dir_move_filt, '--b')
plot(1:length(ind), x_center(1,1).*ones(1, length(ind)), '--r')
plot(1:length(ind), centroid_org_scale_x.*ones(1, length(ind)), '--g')  % to account for the offset, can align centroid of area found with centroid of orignal 
ylabel('x dir: forward (below), back (above)')

subplot(2,1,2)
%plot(y_center(ind,1))  % up (below line) and down (above line)
plot(y_dir_move, 'b')
hold on
plot(y_dir_move_filt, '--b')
plot(1:length(ind), y_center(1,1).*ones(1, length(ind)), '--r')
plot(1:length(ind), centroid_org_scale_y.*ones(1, length(ind)), '--g')  % to account for the offset, can align centroid of area found with centroid of orignal
ylabel('y dir: down (below), up (above)')
xlabel('Number of frames', 'Fontsize', 12)

set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1])
set(gcf,'PaperPositionMode','auto')
set(gcf,'PaperPosition', [1 1 28 19]);    %change the paper position instead of the position
print(gcf, '-dpng', sprintf('%sfig1_xandyVSframes_%s', save_path, subname)) % '-depsc'



figure('Visible', 'on')
subplot(2,1,1)
plot(t, x_center(ind,1))
hold on
plot(t, x_center(1,1).*ones(1, length(t)), '--r')
plot(t, centroid_org_scale_x.*ones(1, length(t)), '--g')
ylabel('x dir: forward (below), back (above)')

subplot(2,1,2)
%plot(t, y_center(ind,1))  % up (below line) and down (above line)
out = (y_center(1,1) - y_center(ind,1)) + y_center(1,1);  % inverse the direction so it is more natural/easy to understand
plot(t, out)
hold on
plot(t, y_center(1,1).*ones(1, length(t)), '--r')
plot(t, centroid_org_scale_y.*ones(1, length(t)), '--g')
ylabel('y dir: down (below), up (above)')
xlabel('Time (sec)', 'Fontsize', 12)

set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1])
set(gcf,'PaperPositionMode','auto')
set(gcf,'PaperPosition', [1 1 28 19]);    %change the paper position instead of the position
print(gcf, '-dpng', sprintf('%sfig2_xandyVStime_%s', save_path, subname)) % '-depsc'



figure('Visible', 'on')
q_measure = abs(area_org_scale - area_found(ind,1));
plot(q_measure)
ylabel('Accuracy: difference in area of org and found')  % the greater the area difference the more erroneous the found region
xlabel('Number of frames', 'Fontsize', 12)

set(gcf, 'units', 'normalized', 'outerposition', [0 0 1 1])
set(gcf,'PaperPositionMode','auto')
set(gcf,'PaperPosition', [1 1 28 19]);    %change the paper position instead of the position
print(gcf, '-dpng', sprintf('%sfig3_areaVSframes_%s', save_path, subname)) % '-depsc'


save(sprintf('%s.mat', out), 'Uroi', '-v7.3');

cd(save_path)

for z = 1:length(out)
    if z == 1
        fileID = fopen(sprintf('%s2.txt', subname), 'w');
    else
        fileID = fopen(sprintf('%s2.txt', subname),  'a+');
    end
    fprintf(fileID, '%d      %f      %f      %f\r\n', z, x_dir_move(z), out(z), q_measure(z));
    fclose(fileID);
end


%%

%rows = size(pixels, 1);     % 240
%cols = size(pixels, 2);     % 320
%rows = size(U, 1);
%cols = size(U, 2);
rows = size(Uroi, 1);
cols = size(Uroi, 2);

g = 0;

for l = 2:nFrames
    l
    % Take the difference between frames
    %d(:,:,l) = abs(pixel(:,:,l) - pixel(:,:,l-1));
    %d(:,:,l) = abs(U(:,:,l) - U(:,:,l-1));
    d(:,:,l) = abs(Uroi(:,:,l) - Uroi(:,:,l-1));
    
    k = d(:,:,l);
    
    % ----------------------------
    figure('Visible', 'on')
    imagesc(k);
    title('Difference between frames')
    colorbar
    %drawnow;
    %himage = imshow(d(:,:,l));
    %hfigure = figure;
    %impixelregionpanel(hfigure, himage);
    %datar = imageinfo(imagesc(d(:,:,l)));
    %disp(datar);
    
    if rem(l,12) == 0
        figure
        for i = 1:12
            subplot(3, 4, i)
            imagesc(d(:,:,i+g));
            if i == 2
                title('Difference between frames')
            end
            drawnow;
            hold on
        end
    end
    % ----------------------------


    % Convert image to binary image, based on threshold (0.2)
    thresh_bw = 0.02;        % ************************* CAN CHANGE
    bw(:,:,l) = im2bw(k, thresh_bw);  
    bw1 = bwlabel(bw(:,:,l));
    
    % ----------------------------
    figure('Visible', 'on')
    imshow(bw(:,:,l))
    title('Convert image to binary image, based on threshold')
    colorbar
    hold on
    
    if rem(l,12) == 0
        figure
        for i = 1:12
            subplot(3, 4, i)
            imshow(bw(:,:,i+g))
            axis image on
            if i == 2
                title('Convert image to binary image, based on threshold')
            end
            drawnow;
            
            hold on
        end
        g = g + 12;
    end
   % ----------------------------
    
   % ----------------------------
   % Modifies orginal grayscale by removing pixels less than a certain
   % threshold - This matrix is NOT used - Sort of useless but could be used
   dsmall = 0.03;        % 0.1 ************************* CAN CHANGE
   for h = 1:rows
       for w = 1:cols
           if(d(:,:,l) < dsmall)
               d(h,w,l) = 0;
           end
       end
   end
   % ----------------------------
  

   % ----------------------------
   % Uses the thresh_edge value to draw a rectangle around detected edges
   % in the black and white image
   cou = 1;
   thresh_edge = 0.3;       %0.5;
   
   for h = 1:rows
       for w = 1:cols
           if(bw(h,w,l) > thresh_edge)
               toplen = h;
               
               if cou == 1
                   tpln = toplen;
               end
               cou = cou + 1;
               break
           end
       end
   end
   
   coun = 1;
   
   for w = 1:cols
       for h = 1:rows
           if (bw(h,w,l) > thresh_edge)
               leftsi = w;
               
               if (coun == 1)
                   lftln = leftsi;
                   coun = coun + 1;
               end
               break
           end
       end
   end
   % ----------------------------
   
   if l > 2
       % --------------------------------------
       % Not needed - useful idea
       % --------------------------------------
       sr = regionprops(bw1, 'BoundingBox');
       s = regionprops(bw1, 'centroid');
       centroids = cat(1, s.Centroid);
       %ang = s.Orientation;
       hold on
       plot(centroids(:,1), centroids(:,2), 'r*')

       for r = 1:length(s)
           rectangle('Position', sr(r).BoundingBox, 'EdgeColor', 'r');
       end
       % --------------------------------------


       % Show a box around the edge it tracks
       figure
       
       imaqmontage(k);      % Displays a sequence of image frames as a montage
       hold on
       
       widh = leftsi - lftln;
       heig = toplen - tpln;
       
       widt = widh/2;
       heit = heig/2;
       
       with = lftln + widt;
       heth = tpln + heit;
       
       wth(l) = with;
       hth(l) = heth;
       
       rectangle('Position', [lftln tpln widh heig], 'EdgeColor', 'r');
       
       plot(with, heth, 'r*');
       title('Image with box based on location of edge')
       drawnow;
   end
   
end



wh  = square(abs(wth(2) - wth(nFrames)));
ht = square(abs(hth(2) - hth(nFrames)));

distan = sqrt(wh + ht);   % Total distance object traveled in the frame (measured from center of rectangle)





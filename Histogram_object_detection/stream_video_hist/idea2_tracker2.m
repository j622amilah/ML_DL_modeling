function var1 = idea2_tracker2(varargin)

U = varargin{1};
org = varargin{2};
rect = varargin{3};
rect2 = varargin{4};
var1 = varargin{5};

b = 0;
c_row = 1;


for picrows = 1:var1.dstep_row:(rect(1, 4) - var1.a)
    prs = picrows;
    prf = var1.a + picrows-1;
    c_across = 1;
    % slide feature across original image like convolution - checking for a similarity using
    % translation - not good for rotations of the same feature (need a different way to quantify a feature in space)

    for picacross = 1:var1.dstep_acr:(rect(1, 3) - var1.aw)
        
        ccc = var1.aw + picacross-1;
        
        dmatch{c_row, c_across} = abs(org - U(prs:prf, picacross:ccc, var1.num_of_frames));
        
        fee = imhist(org);
        
        sect = imhist(U(prs:prf, picacross:ccc, var1.num_of_frames));
        
        histout{c_row, c_across} = abs(fee - sect);
        
        histoutmean(c_row, c_across) = mean(histout{c_row, c_across});

        qq(c_row, c_across) = mean(mean(dmatch{c_row, c_across}));

        c_across = c_across + 1;
    end

    b = b + 1;
    c_row = c_row + 1;
end



% Works well if feature is small: Check to see if current frame is similar to first frame - skip (or do another type of analysis to track) frames where lighting dramatically changes

if var1.num_of_frames == 1
    var1.histoutmean_1store = histoutmean;
end
histoutmean_diff = abs(var1.histoutmean_1store - histoutmean);
mean(mean(histoutmean_diff))

if mean(mean(histoutmean_diff)) < var1.algoOpt1     %0.85  %0.13    % if lighting changes or if feature object is too big; hist is very different than initial hist - cut off value needs to be large if you want to view tracking, signifies poor quality tracking  
    

    % --------------- PRINTING ---------------
    axes(var1.det_ax)   % select x direction movement axes
    var1.det_ax_loc = imagesc(histoutmean);
    % -----------------------------------------

    
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

    
    % --------------- PRINTING ---------------
    axes(var1.det_ax)   % select x direction movement axes
    var1.det_ax_loc = imagesc(histoutmean_bin);
    % -----------------------------------------
    
    
    
    % ---------------------------------------------------------------------------
    % ---------------
    % Remove false detection of pixels not belonging to the main/largest group of detected pixels
    % ---------------
    if var1.run_falsedetection_removal == 1
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
    end
    % ---------------------------------------------------------------------------



    % Ensure that the size of the largest group is equivalent to the size of the orginal goal/feature image
    sr = regionprops(histoutmean_bin, 'BoundingBox');

    s = regionprops(histoutmean_bin, 'centroid');
    centroids = cat(1, s.Centroid);
    % **************************
    % Get measure of error/reliability by comparing with original position and size of "feature" image
    scale_xmin = ((length(1:var1.dstep_acr:var1.cols))*rect2(1,1))/rect(1, 3);
    scale_ymin = ((length(1:var1.dstep_row:var1.rows))*rect2(1,2))/rect(1, 4);

    scalefactor_x = round((rect(1, 3)-var1.aw-1)/rect2(1, 3));
    scalefactor_y = round((rect(1, 4)-var1.a-1)/rect2(1, 4));
    scalefactor_x2 = ((length(1:var1.dstep_acr:(rect(1, 3)-var1.a)))*rect2(1,3))/rect(1, 3);     % original size
    scalefactor_y2 = ((length(1:var1.dstep_row:(rect(1, 4)-var1.aw)))*rect2(1,4))/rect(1, 4);

    area_org_scale = scalefactor_x*scalefactor_y;
    var1.area_found(var1.num_of_frames,1) = sr(1).BoundingBox(1,3)*sr(1).BoundingBox(1,4);
    centroid_org_scale_x = scale_xmin+(scalefactor_x/2);
    centroid_org_scale_y = scale_ymin+(scalefactor_y/2);

    var1.x_center(var1.num_of_frames,1) = centroids(:,1);
    var1.y_center(var1.num_of_frames,1) = centroids(:,2);
    % **************************
    
    
    % --------------- PRINTING ---------------
    axes(var1.det_ax)   % select x direction movement axes
    var1.det_ax_loc = imagesc(histoutmean_bin);
    hold on
    plot(centroids(:,1), centroids(:,2), 'r*')
    plot(centroid_org_scale_x, centroid_org_scale_y, 'g*')
    rectangle('Position', sr(1).BoundingBox, 'EdgeColor', 'r');
    rectangle('Position', [scale_xmin, scale_ymin, scalefactor_x, scalefactor_y], 'EdgeColor', 'g');
    rectangle('Position', [scale_xmin, scale_ymin, scalefactor_x2, scalefactor_y2], 'EdgeColor', 'm');
    % -----------------------------------------
    
else
    var1.x_center(var1.num_of_frames, 1) = 0;
    var1.y_center(var1.num_of_frames, 1) = 0;

    var1.area_found(var1.num_of_frames, 1) = 100;      % big number to represent that there is no value

end     % end of if mean(mean(histoutmean_diff)) < 0.13
        
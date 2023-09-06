
%             % Start of tracking algorithm
%             b = 0;
%             c_row = 1;
%             
%             
%             % initialize variables
%             dstep_row = var1.dstep_row;
%             dstep_acr = var1.dstep_acr;
%             h_small = Uroi_h_small;
%             w_small = Uroi_w_small;
%             rows = var1.rows;
%             cols = var1.cols;
%             a = var1.a;
%             aw = var1.aw;
%             U = Uroi;
%             %org = org;
%             num_of_frames = var1.num_of_frames;

           
            
            %sprintf('%d      %d       %d       %d       %d       %d       %d       %d\n\r', dstep_row, dstep_acr, h_small, w_small, rows, cols, a, aw)
%             
%             for picrows = 1:dstep_row:(h_small-a)
%                 prs = picrows;
%                 prf = a + picrows-1;
%                 c_across = 1;
%                 % slide feature across original image like convolution - checking for a similarity using
%                 % translation - not good for rotations of the same feature (need a different way to quantify a feature in space)
% 
%                 for picacross = 1:dstep_acr:(w_small-aw)
% 
%                     picacross
% 
%                     ccc = aw + picacross-1;
% 
%                     dmatch{c_row, c_across} = abs(org - U(prs:prf, picacross:ccc, num_of_frames));
% 
%                     fprintf('dmatch finished\n\r')
% 
%                     fee = imhist(org);
%                     fprintf('fee finished\n\r')
% 
%                     sect = imhist(U(prs:prf, picacross:ccc, num_of_frames)); 
%                     fprintf('sect finished\n\r')
% 
%                     histout{c_row, c_across} = abs(fee - sect);
%                     fprintf('histout finished\n\r')
% 
%                     histoutmean(c_row, c_across) = mean(histout{c_row, c_across});
%                     fprintf('histoutmean finished\n\r')
% 
%                     qq(c_row, c_across) = mean(mean(dmatch{c_row, c_across}));
% 
%                     %[mssim(c_row, c_across) ssim_map] = ssim_index(org, U(prs:prf, picacross:ccc, f));  % similarity
% 
%                     c_across = c_across + 1;
%                 end
% 
%                 histoutmean
% 
%                 b = b + 1;
%                 c_row = c_row + 1;
%             end
% 
%             fprintf('HERE Finished\n\r')
% 
% % ---------------------------------------------------
% % Check to see if current frame is similar to first frame - skip (or do another type of analysis to track) frames where lighting dramatically changes
% if num_of_frames == 1
%     histoutmean_1store = histoutmean;
% end
% histoutmean_diff = abs(histoutmean_1store - histoutmean);
% 
% if mean(mean(histoutmean_diff)) < 0.13
%     % Perform tracking
% 
%     % --------------- PRINTING ---------------
% %     if rem(num_of_frames, 20)
% %         figure
% %         imagesc(histoutmean)
% %         colorbar
% %     end
% % ---------------
% 
%     [r1, r2] = size(histoutmean);
% 
%     % detect pixels that are strongly similar to the goal/feature image
%     [y,i] = sort(min(histoutmean));
%     num = floor(length(y)*0.1);         % weird error with floor 25/01/17 - replace function if it happens again %num = length(y)*0.1
%     bin_threshhold = y(1,num);
%     [ro, co] = find(histoutmean < bin_threshhold);
%     histoutmean_bin = zeros(r1,r2);
%     for i = 1:length(ro)
%         histoutmean_bin(ro(i),co(i)) = 1;
%     end
% 
%     %figure
%     %imagesc(histoutmean_bin)
%     %colorbar
% 
%     % Remove false detection of pixel not belonging to the main/largest group of detected pixels
%     [y, ind] = sort(co); %#ok<ASGLU>
%     for j = 1:length(ind)
%         co1(j,1) = co(ind(j,1),1);
%         ro1(j,1) = ro(ind(j,1),1);
%     end
%     clear y ind co
% 
%     % determine the number of groups of detected pixels 
%     grop_int = 1;
%     grop(1,1) = 1;
%     cc = 1;
%     rr_grop_before = ro1(1,1);
% 
%     for i = 1:length(ro1)-1
%        if (co1(i+1,1) == co1(i,1)+1) || (co1(i+1,1) == co1(i,1)-1) || (co1(i+1,1) == co1(i,1))
%            mean_rr_grop = mean([rr_grop_before; ro1(i+1,1)]);
%            rr_grop_before = [rr_grop_before; ro1(i+1,1)];
% 
%            if abs(ro1(i+1,1)-mean_rr_grop) < 4
%                grop(i+1,1) = grop_int;
%                cc = cc + 1;
%            else
%                rr_grop_before = ro1(i+1,1);
% 
%                grop_int = grop_int + 1;
%                grop(i+1,1) = grop_int;
%                cc = 1;
%            end
%        else
%            rr_grop_before = ro1(i+1,1);
% 
%            grop_int = grop_int + 1;
%            grop(i+1,1) = grop_int;
%            cc = 1;
%        end
%     end
% 
%     [vals, loc, ci] = unique(grop);
% 
%     grop_s(1,1) = loc(1);
%     for i = 2:length(vals)
%         grop_s(i,1) = loc(i)-loc(i-1);
%     end
% 
%     % only keep the largest group of detected pixels
%     [y, b_grop_loc] = max(grop_s);
%     clear grop_s % y grop_s vals ci grop
% 
%     % get new co1 and ro1 
%     if b_grop_loc == 1
%         co1_bgrop = co1(1:loc(b_grop_loc));
%         ro1_bgrop = ro1(1:loc(b_grop_loc));
%     else
%         co1_bgrop = co1((loc(b_grop_loc-1)+1):loc(b_grop_loc));
%         ro1_bgrop = ro1((loc(b_grop_loc-1)+1):loc(b_grop_loc));
%     end
% 
%     %clear b_grop_loc co1 ro1
% 
%     % assign detected pixels from largest group
%     for i = 1:length(ro1_bgrop)
%         histoutmean_bin(ro1_bgrop(i),co1_bgrop(i)) = 1;
%     end
% 
%     % Ensure that the size of the largest group is equivalent to the size of the orginal goal/feature image
%     sr = regionprops(histoutmean_bin, 'BoundingBox');
% 
%     s = regionprops(histoutmean_bin, 'centroid');
%     centroids = cat(1, s.Centroid);
%     % **************************
%     % Get measure of error/reliability by comparing with original position and size of "feature" image
%     %scale_xmin = ((length(1:dstep_acr:(w_small-a)))*rect2(1,1))/rect(1, 3);
%     %scale_ymin = ((length(1:dstep_row:(h_small-aw)))*rect2(1,2))/rect(1, 4);
%     scale_xmin = ((length(1:dstep_acr:cols))*rect2(1,1))/rect(1, 3);
%     scale_ymin = ((length(1:dstep_row:rows))*rect2(1,2))/rect(1, 4);
% 
%     scalefactor_x = round((w_small-aw-1)/rect2(1, 3));
%     scalefactor_y = round((h_small-a-1)/rect2(1, 4));
%     scalefactor_x2 = ((length(1:dstep_acr:(w_small-a)))*rect2(1,3))/rect(1, 3);     % original size
%     scalefactor_y2 = ((length(1:dstep_row:(h_small-aw)))*rect2(1,4))/rect(1, 4);
% 
%     area_org_scale = scalefactor_x*scalefactor_y;
%     area_found(num_of_frames,1) = sr(1).BoundingBox(1,3)*sr(1).BoundingBox(1,4);
%     centroid_org_scale_x = scale_xmin+(scalefactor_x/2);
%     centroid_org_scale_y = scale_ymin+(scalefactor_y/2);
% 
%     x_center(num_of_frames,1) = centroids(:,1);
%     y_center(num_of_frames,1) = centroids(:,2);
%     % **************************
% 
%     % --------------- PRINTING ---------------
% %     if rem(num_of_frames, 20)
% %         figure
% %         imagesc(histoutmean_bin)
% %         colorbar
% %         hold on
% %         plot(centroids(:,1), centroids(:,2), 'r*')
% %         plot(centroid_org_scale_x, centroid_org_scale_y, 'g*')
% % 
% %         rectangle('Position', sr(1).BoundingBox, 'EdgeColor', 'r');
% %         rectangle('Position', [scale_xmin, scale_ymin, scalefactor_x, scalefactor_y], 'EdgeColor', 'g');
% %         rectangle('Position', [scale_xmin, scale_ymin, scalefactor_x2, scalefactor_y2], 'EdgeColor', 'm');
% %     end
% % ---------------
% 
% 
% 
%     if rem(num_of_frames,100) == 0
%         close all 
%     end
%     % ---------------------------------------------------
% 
% else
%     x_center(num_of_frames,1) = 0;
%     y_center(num_of_frames,1) = 0;
% 
%     area_found(num_of_frames,1) = 100;      % big number to represent that there is no value
% 
% end     % end of if mean(mean(histoutmean_diff)) < 0.13
%             


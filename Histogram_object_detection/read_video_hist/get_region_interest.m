function [svar, smats] = get_region_interest(U, svar, current_frame_num)

% 3c. Region of interest
% ++++++++++++++++++++ Plotting ++++++++++++++++++++
figure
imshow(U, []);
title('Select the region in which you are interested in:')
svar.rect = getrect;
close all
% +++++++++++++++++++++++++++++++++++++++++++++++++++

smats = asign_U_to_Uroi(U, svar, current_frame_num);

svar.h_small = svar.rect(1, 4);
svar.w_small = svar.rect(1, 3);
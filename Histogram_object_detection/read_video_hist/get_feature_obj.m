function svar = get_feature_obj(svar, smats)

% ----------------------------------------
% Idea 2: find a unique feature to find in the image
% ----------------------------------------
svar.rows = size(smats.Uroi, 1);
svar.cols = size(smats.Uroi, 2);

% ++++++++++++++++++++ Plotting ++++++++++++++++++++
figure
imshow(smats.Uroi(:, :, 1), []);
title('Select a search object:')
svar.rect2 = getrect;
close all
% +++++++++++++++++++++++++++++++++++++++++++++++++++

x_min = round(svar.rect2(1,2));
y_min = round(svar.rect2(1,1));
if x_min > 1
    x_vec2 = x_min:(round(svar.rect2(1,4))+x_min-1);
else
    x_vec2 = x_min:round(svar.rect2(1,4));
end

if y_min > 1
    y_vec2 = y_min:(round(svar.rect2(1,3))+y_min-1);
else
    y_vec2 = y_min:round(svar.rect2(1,3));
end

svar.org = smats.Uroi(x_vec2, y_vec2, 1);

% ++++++++++++++++++++ Plotting ++++++++++++++++++++
% figure
% imshow(org)
% title('Search object')
% +++++++++++++++++++++++++++++++++++++++++++++++++++

svar.a = size(svar.org, 1); 
svar.aw = size(svar.org, 2);
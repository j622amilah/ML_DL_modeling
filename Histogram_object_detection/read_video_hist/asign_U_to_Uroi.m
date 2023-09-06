function smats = asign_U_to_Uroi(U, svar, current_frame_num)

x_min = round(svar.rect(1,2));
y_min = round(svar.rect(1,1));
if x_min > 1
    x_vec = x_min:(round(svar.rect(1,4))+x_min-1);
else
    x_vec = x_min:round(svar.rect(1,4));
end

if y_min > 1
    y_vec = y_min:(round(svar.rect(1,3))+y_min-1);
else
    y_vec = y_min:round(svar.rect(1,3));
end

smats.Uroi(:,:,current_frame_num) = U(x_vec, y_vec);


clear U
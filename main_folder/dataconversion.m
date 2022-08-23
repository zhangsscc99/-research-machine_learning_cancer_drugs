

load('KS_score_2017002004_2017002005.mat'); 
A=combined_ks_all_time(:,:,1);

csvwrite('A.csv', combined_ks_all_time);

function [tmrse_vec] = compute_trmse(signal_desired, signal_predicted, target_radius)
squared_error = (signal_desired - signal_predicted).^2;
se_tresh = target_radius^2;
thresholded_se = wthresh(squared_error, 's', se_tresh);
tmrse_vec = sqrt(mean(thresholded_se));
end
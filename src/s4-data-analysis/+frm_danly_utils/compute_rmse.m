function [rmse_vec] = compute_rmse(signal_desired, signal_predicted)
rmse_vec = rmse(signal_desired, signal_predicted);
end
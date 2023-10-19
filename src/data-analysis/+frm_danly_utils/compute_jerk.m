function [jerk_vec] = compute_jerk(signal_predicted, sample_rate)
dt = 1/sample_rate;
pos = signal_predicted;
vel = diff(pos)/dt;
acc = diff(vel)/dt;
jerk = diff(acc)/dt;
lmaj = log10(mean(abs(jerk))+1); % shift to account for zero jerk
% thresholded_lmaj = extractdata(relu(dlarray(lmaj)));
% jerk_vec = 3*log10(sample_rate) +  thresholded_lmaj;
jerk_vec = 3*log10(sample_rate) +  lmaj;
end
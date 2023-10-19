function [max_h] = compute_maxh(signal_desired, signal_predicted, target_radius, sample_rate)
squared_error = (signal_desired - signal_predicted).^2;
se_thresh = target_radius^2;
thresholded_se = wthresh(squared_error, 's', se_thresh);

cumulative_tse = cumsum(thresholded_se);
max_tse_count = max(cumulative_tse);
num_dofs = size(thresholded_se, 2);
max_h = zeros(1, num_dofs);
for chan = 1:num_dofs
    chan_max_h = 0;
    for i = max_tse_count(chan):-1:1
        chan_max_h = max(chan_max_h, sum(cumulative_tse(:,chan)==i));
    end
    max_h(chan) = chan_max_h;
end
max_h = max_h-1; % correct off-by-1 bias
max_h = max_h * 1/sample_rate;
end


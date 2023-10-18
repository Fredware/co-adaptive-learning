function [ptit_vec] = compute_ptit(val_desired, val_predicted, se_thresh)
squared_error = (val_desired - val_predicted).^2;
thresholded_se = wthresh(squared_error, 's', se_thresh);
ptit_vec = sum(thresholded_se==0)./length(thresholded_se)*100;
end
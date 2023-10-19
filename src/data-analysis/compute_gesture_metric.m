function weighted_metric = compute_gesture_metric(input_ttbl, metric_type, gesture_name)
gesture_rows = input_ttbl.("Gesture") == gesture_name;
kin_des = input_ttbl{gesture_rows, "Targets"};
kin_pred = input_ttbl{gesture_rows, "Kalman"};
target_radius = 0.10;
sample_rate = 1/33e-3;
switch metric_type
    case "trmse"
        metric_vals = frm_danly_utils.compute_trmse(kin_des, kin_pred, target_radius);
    case "lmaj"
        metric_vals = frm_danly_utils.compute_jerk(kin_pred, sample_rate);
    case "ptit"
        metric_vals = frm_danly_utils.compute_ptit(kin_des, kin_pred, target_radius);
end
weighted_metric = mean( [mean(metric_vals(1:3)), mean(metric_vals(4:5))]);
end
function [segmented_ttbl] = extract_segment(input_ttbl, segmentation_value, task_type)
kalman_variable = "Kalman";
if (task_type == "ml")
    kalman_variable = join(["Kalman",  regexp(segmentation_value, "(\d{2})", 'match')]);
    segmentation_value = "Trial 09";
end
segmented_rows = input_ttbl.("Trial") == segmentation_value;
segmented_ttbl = input_ttbl(segmented_rows, ["Gesture", "Targets", kalman_variable]);
if (task_type == "ml")
    segmented_ttbl = renamevars(segmented_ttbl, kalman_variable, "Kalman");
end
end
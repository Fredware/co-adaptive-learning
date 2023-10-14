function output_ttbl = prune_timetable(input_ttbl, task_type)
%REDUCE_TIMETABLE Discard unused DOFs and Feature channels
%   Detailed explanation goes here
reduced_dofs = [1:5];
reduced_feats = [193:224];

target_array = input_ttbl.("Targets");
features_array = input_ttbl.("Features");
output_ttbl = timetable(input_ttbl.("NIP Time"), target_array(:, reduced_dofs), features_array(:,reduced_feats),'RowTimes', input_ttbl.Time, 'VariableNames', ["NIP Time", "Targets", "Features"]);
if task_type == "ml"
    for i = 1:10
        var_name = sprintf("Kalman %02d", i);
        kalman_array = input_ttbl.(var_name);
        output_ttbl = addvars(output_ttbl, kalman_array(:, reduced_dofs), 'NewVariableNames', [var_name]);
    end
else
    kalman_array =  input_ttbl.("Kalman");
    output_ttbl = addvars(output_ttbl, kalman_array(:, reduced_dofs), 'NewVariableNames', ["Kalman"] );
end
kinematics_array = input_ttbl.("Kinematics");
output_ttbl = addvars(output_ttbl, kinematics_array(:, reduced_dofs), 'NewVariableNames', ["Kinematics"]);
end
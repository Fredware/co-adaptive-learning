function y = compute_performance_table(participant_array, metric_type, metric_task)
y = [];
trials = (1:9)';
for i = 1:numel(participant_array)
    participant_handle = participant_array(i);
    fprintf("Participant %02d: %s\n", i, participant_handle.participant_id)
    cond_dict = participant_handle.cond_dict;
    cond_keys = keys(cond_dict);
    y_all_conds= [];
    for j = 1:numEntries(cond_dict)
        cond_label = cond_keys(j);
        fprintf("\tCondition: %s\n", cond_label)
        task_dict = cond_dict(cond_label).task_dict;
        task_keys = keys(task_dict);
        y_all_tasks = [];
        for k = 1:numEntries(task_dict)
            task_label = task_keys(k);
            fprintf("\t\tTask: %s\n", task_label)
            task_ttbl = task_dict{task_label}.data_ttbl;
            y_metric_all_trials = [];
            for t = 1:numel(trials)
                trial_label = sprintf("Trial %02d", t);
                trial_ttbl = extract_segment(task_ttbl, trial_label, task_label);
                trmse_coact = compute_gesture_metric(trial_ttbl, metric_type, "Co-Activation");
                trmse_diffact = compute_gesture_metric(trial_ttbl, metric_type, "Differentiation");
                fprintf("\t\t\t%s\tCo-Activation:%.3f\t Differentiation:%.3f\n", trial_label, trmse_coact, trmse_diffact)
                y_metric_all_trials = [y_metric_all_trials; [trmse_coact, trmse_diffact]];
            end
            y_task = compute_y_task(y_metric_all_trials, metric_task);
            y_all_tasks = [y_all_tasks, y_task];
        end
        y_cond = [table(cond_label, 'VariableNames', "b_condition"), array2table(y_all_tasks, "VariableNames",arrayfun(@(x) sprintf("y_%d", x), 1:size(y_all_tasks, 2)))];
        y_all_conds = [y_all_conds; y_cond];
    end
    y = [y; y_all_conds];
end
end
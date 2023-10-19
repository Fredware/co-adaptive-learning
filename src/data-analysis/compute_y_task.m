function y_task = compute_y_task(y_metric_all_trials, computation_mode)
trials = (1:size(y_metric_all_trials,1))';
switch computation_mode
    case "learning effect"
        y_task = y_metric_all_trials(6:9,:) - y_metric_all_trials(1:4,:); %final - initial
        y_task = [y_task(:,1)', y_task(:,2)']; % coact x4 then diffact x4
    case "exponential learning rate"
        for l=1:size(y_metric_all_trials,2)
            exp_model{l} = fit(trials, y_metric_all_trials(:, l), 'exp1', 'StartPoint', [0.6, -0.2]); % Initial guess from x=1,9;y = 0.5,0.1
            exp_coeffs(l,:) = coeffvalues(exp_model{l});
        end
        y_task = [exp_coeffs(1,2), exp_coeffs(2,2)];
    case "linear learning rate"
        for l=1:size(y_metric_all_trials,2)
            lin_model{l} = fit(trials, y_metric_all_trials(:, l), 'poly1');
            lin_coeffs(l,:) = coeffvalues(lin_model{l});
        end
        y_task = [lin_coeffs(1,1), lin_coeffs(2,1)];
    otherwise
        y_task = [y_metric_all_trials(:,1)', y_metric_all_trials(:,2)'];
end
end
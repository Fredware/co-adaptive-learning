function output_ttbl = label_timetable(input_ttbl, input_events_tbl, task_type)
%LABEL_TIMETABLE Add Trial and Gesture columns based on the events_tbl
%   Detailed explanation goes here

nip_time = input_ttbl.("NIP Time");
input_targets = input_ttbl.("Targets");
trial_labels = repmat("NA", size(nip_time));
gesture_labels = repmat("Rest", size(nip_time));
if task_type == "ml"
    trial_id = 0;
    num_trials = 10;
else
    trial_id = 1;
    num_trials = 9;
end
coact_starts = input_events_tbl.TargOnTS(1:3:3*num_trials);
coact_stops = input_events_tbl.TrialTS(1:3:3*num_trials);
diffact_starts = input_events_tbl.TargOnTS(2:3:3*num_trials);
diffact_stops = input_events_tbl.TrialTS(2:3:3*num_trials);
if isa(coact_starts, 'cell'); coact_starts = [coact_starts{:}]'; end
if isa(coact_stops, 'cell'); coact_stops = [coact_stops{:}]'; end
if isa(diffact_starts, 'cell'); diffact_starts = [diffact_starts{:}]'; end
if isa(diffact_stops, 'cell'); diffact_stops = [coact_stops{:}]'; end
% if or(task_type == "cl", task_type == "hl")
%     coact_starts = [coact_starts{:}]';
%     diffact_starts = [diffact_starts{:}]';
% end

for i = 1:num_trials
    gesture_range = and(nip_time >= coact_starts(i), nip_time<=coact_stops(i));
    target_range = and(gesture_range, input_targets(:, 1) ~=0);
    gesture_labels(target_range) = "Co-Activation";
    gesture_range = and(nip_time >= diffact_starts(i), nip_time<=diffact_stops(i));
    target_range = and(gesture_range, input_targets(:, 1) ~=0);
    gesture_labels(target_range) = "Differentiation";
    trial_range = and(nip_time >= coact_starts(i), nip_time<=diffact_stops(i));
    trial_labels(trial_range) = sprintf("Trial %02d", trial_id);
    trial_id = trial_id +1;
end
output_ttbl = addvars(input_ttbl, trial_labels, gesture_labels, 'Before','Targets', 'NewVariableNames',["Trial", "Gesture"]);
end
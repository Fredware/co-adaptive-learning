classdef TrainingTask < Task
    %TRAININGTASK Summary of this class goes here
    %   Detailed explanation goes here

    properties
        % kalman_gain
    end

    methods
        function obj = TrainingTask(data_dir)
            %TRAININGTASK Construct an instance of this class
            %   Detailed explanation goes here
            file_list = dir(data_dir);
            expression = '^TrainingData_(?!Initial)[\w\-]*\.kdf$';
            matching_files = file_list(~cellfun('isempty', regexp({file_list.name}, expression, "once")));
            if numel(matching_files) == 1
                task_file = fullfile(data_dir, matching_files(1).name);
            else
                error("Error finding Task files in:\n\t%s\nFound %d files, expected 1.", data_dir, numel(matching_files));
            end
            fprintf("Loading Training File: %s\n", task_file);
            [kinematics, features, targets, kalman, nip_timestamps] = jag_deng_utils.read_kdf_file(task_file);
            training_data_ttbl = timetable(nip_timestamps', targets', features', kalman', kinematics', 'SampleRate', 1/33e-3);
            training_data_ttbl.Properties.VariableNames = ["NIP Time", "Targets", "Features", "Kalman", "Kinematics"];

            events_file = regexprep(task_file, '\.kdf$', '.kef');
            obj.events_tbl = struct2table(jag_deng_utils.read_kef_file(events_file));

            trial_nip_stops = obj.events_tbl{2:3:end, "TrialTS"}; % initial event=2, events per trial=3

            for i=1:1:length(trial_nip_stops)
                nip_stop = trial_nip_stops(i);
                nip_training_range = training_data_ttbl.("NIP Time") <= nip_stop;
                training_set_ttbl = training_data_ttbl(nip_training_range, :);
                unwanted_channels = [1:192]; % 3 USEAs x 64 channels = 192
                fprintf("Training Kalman: %02d iteration(s)\n", i);
                [training_kinematics, training_features] = jag_deng_utils.align_training_data(training_set_ttbl.("Kinematics")', training_set_ttbl.("Features")', unwanted_channels, 'standard');
                feature_idxs = jag_deng_utils.select_chans(training_kinematics, training_features, unwanted_channels, 'gramSchmDarpa', 48); % 48: legacy from Nieveen et al 2017
                kalman_filter = jag_deng_utils.train_decoder(training_kinematics, training_features, feature_idxs, 'standard');
                kalman_output = jag_deng_utils.run_decoder(kalman_filter, training_data_ttbl.("Kinematics")', training_data_ttbl.("Features")', feature_idxs, 'standard');
                if i==1
                    obj.data_ttbl = timetable(training_data_ttbl.("NIP Time"), ...
                        training_data_ttbl.("Targets"), ...
                        training_data_ttbl.("Features"), ...
                        kalman_output', 'SampleRate', 1/33e-3);
                    obj.data_ttbl.Properties.VariableNames = ["NIP Time", "Targets", "Features", "Kalman 01"];
                else
                    obj.data_ttbl = addvars(obj.data_ttbl, kalman_output', 'NewVariableNames', sprintf("Kalman %02d", i));
                end
            end
            obj.data_ttbl = addvars(obj.data_ttbl, training_data_ttbl.("Kinematics"), 'NewVariableNames', "Kinematics");
        end

        function outputArg = method1(obj,inputArg)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            outputArg = obj.Property1 + inputArg;
        end
    end
end


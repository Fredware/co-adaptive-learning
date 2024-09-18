classdef ControlTask < Task
    %UNTITLED4 Summary of this class goes here
    %   Detailed explanation goes here

    methods
        function obj = ControlTask(data_dir)
            %UNTITLED4 Construct an instance of this class
            %   Detailed explanation goes here
            file_list = dir(data_dir);
            
            expression = '^Task[\w\-]*\.kdf$';
            matching_files = file_list(~cellfun('isempty', regexp({file_list.name}, expression, "once")));
            if numel(matching_files) == 1
                task_file = fullfile(data_dir, matching_files(1).name);
            else
                error("Error finding Task files in:\n\t%s\nFound %d files, expected 1.", data_dir, numel(matching_files));
            end
            fprintf("Loading Task File: %s\n", task_file);
            [kinematics, features, targets, kalman, nip_timestamps] = jag_deng_utils.read_kdf_file(task_file);
            obj.data_ttbl = timetable(nip_timestamps', targets', features', kalman', kinematics', 'SampleRate', 1/33e-3);
            obj.data_ttbl.Properties.VariableNames = ["NIP Time", "Targets", "Features", "Kalman", "Kinematics"];

            expression = '^EventParams[\w\-]*\.txt$';
            matching_files = file_list(~cellfun('isempty', regexp({file_list.name}, expression, "once")));
            if numel(matching_files) == 1
                events_file = fullfile(data_dir, matching_files(1).name);
            else
                error("Error finding Events file in:\n\t%s\nFound %d files, expected 1.", data_dir, numel(matching_files));
            end
            [trials_struct, ~] = jag_deng_utils.read_event_params(events_file);
            obj.events_tbl = struct2table(trials_struct);
            obj.events_tbl = obj.events_tbl(:, ["TargOnTS", "TrialTS", "MvntMat"]);
        end

        function outputArg = method1(obj,inputArg)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            outputArg = obj.Property1 + inputArg;
        end
    end
end
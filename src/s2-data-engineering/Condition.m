classdef Condition
    %CONDITION Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        task_dict
    end
    
    methods
        function obj = Condition(data_dir)
            %CONDITION Construct an instance of this class
            %   Detailed explanation goes here
            dict_keys = ["ml", "hl", "cl"];
            dict_vals = {TrainingTask(fullfile(data_dir, 'ml')), ...
                ControlTask(fullfile(data_dir, 'hl')), ...
                ControlTask(fullfile(data_dir, 'cl'))};
            obj.task_dict = dictionary(dict_keys, dict_vals);
        end
        
        function outputArg = method1(obj,inputArg)
            %METHOD1 Summary of this method goes here
            %   Detailed explanation goes here
            outputArg = obj.Property1 + inputArg;
        end
    end
end


classdef StrokeParticipant < Participant
    %UNTITLED Summary of this class goes here
    %   Detailed explanation goes here

    methods
        function obj = StrokeParticipant(data_dir, id_str)
            %UNTITLED Construct an instance of this class
            %   Detailed explanation goes here
            obj.participant_id = id_str;
            dict_keys = ["par", "xpar"];
            dict_vals = [Condition(fullfile(data_dir, 'paretic')), ... 
                Condition(fullfile(data_dir, 'nonparetic'))];
            obj.cond_dict = dictionary(dict_keys, dict_vals);
        end
    end
end


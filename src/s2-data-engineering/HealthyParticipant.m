classdef HealthyParticipant < Participant
    %HEALTHYPARTICIPANT Summary of this class goes here
    %   Detailed explanation goes here

    methods
        function obj = HealthyParticipant(data_dir, id_str)
            %HEALTHYPARTICIPANT Construct an instance of this class
            %   Detailed explanation goes here
            obj.participant_id = id_str;
            dict_keys = ["hlt"];
            dict_vals = [Condition(fullfile(data_dir, 'healthy'))];
            obj.cond_dict = dictionary(dict_keys, dict_vals);
        end
    end
end


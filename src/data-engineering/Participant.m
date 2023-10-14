classdef Participant < matlab.mixin.Heterogeneous
    %PARTICIPANT Class containing all data from a participant
    %   Detailed explanation goes here
    
    properties
        participant_id
        cond_dict
    end
    % 
    % methods (Abstract)
    %     reduce_data_ttbl(obj, dof_idxs, feature_idxs)
    % end
end
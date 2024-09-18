function [XAligned, ZAligned, varargout] = align_training_data(XOrig, ZOrig, badIdxs, method)
% inputs
%   Xorig: 12 x n samples kinematic data movement cue
%   Zorig: 720 x n samples Feature data
%   BadChans: vector of bad chan idx
%   method: string 'standard' (correlation based) or 'trialByTrial' 
% outputs:
% XAligned: 12 x n aligned kinematics
% ZAligned: 720 x n aligned features
% smw

% init
XAligned = XOrig;
ZAligned = ZOrig;

TrainZ(badIdxs, :) = 0;
switch method
    case 'standard'
        % find lag, apply to training data
        [Mvnts,Idxs,MaxLag,~,C] = jag_deng_utils.select_corr_move_chans(XOrig,ZOrig,0.4,badIdxs);
        ZAligned = circshift(ZOrig, MaxLag,2);
        varargout{1} = MaxLag;
        varargout{2} = C;
    case 'trialByTrial'
        [XAligned,ZAligned] = jag_deng_utils.realign_iter_combo(XOrig,ZOrig);%
        % note: could zap badKalmanIdxs at this point before sending to train
        
end
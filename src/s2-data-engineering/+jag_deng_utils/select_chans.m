function Idxs = select_chans(TrainX, TrainZ, badIdxs, method, varargin)
% inputs:
%   TrainX: 12 x n samples kinematic data
%   TrainZ: 720 x n samples Feature data
%   BadIdxs: vector of bad chan idx
%   method: string 'autoSelectMvntsChsCorr_FD' ,
%   ''AkaikeGramSchmChanSelv300_orig05112016', 'gramSchmDarpa' , 'larsDarpa' 
% outputs:
%   Idxs: integer vector of selected indices
%
% smw 3/2017
% based on: C:\Users\Marta Iversen\Downloads\Scripts\selectChans_jag.m

maxChans = 720; % default 
if nargin > 3
    maxChans = varargin{1};
end

switch method
    case 'autoSelectMvntsChsCorr_FD' % autocorr
        [Mvnts,Idxs,MaxLag] = jag_deng_utils.select_corr_move_chans(TrainX,TrainZ,0.4,badIdxs); %if KalmanType is ReFit, run in velocity mode
        Idxs = setdiff(Idxs, badIdxs);
     
    case 'AkaikeGramSchmChanSelv300_orig05112016' % gramscmidt original
        error('DEPRECATED')
    % DEPRECATED. 
    % Ask T. Davis for file: C:\Users\Tyler\Dropbox\Work\Code\Tasks\FeedbackDecode\resources\AkaikeGramSchmChanSelv300_orig05112016.m    
    %     IdxsCell = AkaikeGramSchmChanSelv300_orig05112016(TrainX,TrainZ,[1:size(TrainX,1)],maxChans);
    %     Idxs = unique(cell2mat(IdxsCell));
    %     Idxs = setdiff(Idxs, badIdxs);
    case 'gramSchmDarpa'   % gramScmidt new 8/26/2016 (params?)
        %             IdxsCell = gramSchmDarpa(TrainX,TrainZ,[1:length(KalmanMvnts)],floor(1*720), 0, 'none', 'all');
        IdxsCell = jag_deng_utils.select_gram_schm_darpa(TrainX,TrainZ,[1:size(TrainX,1)],maxChans, 0);
        Idxs = unique(cell2mat(IdxsCell));
        %         Idxs = setdiff(Idxs, badIdxs);
   
    case 'larsDarpa' % lars
        error('DEPRECATED')
    % DEPRECATED
    % Ask T. Davis for file: C:\Users\Tyler\Dropbox\Work\Code\Tasks\FeedbackDecode\resources\larsDarpa.m
    %     IdxsCell = larsDarpa(TrainX,TrainZ,[1:size(TrainX,1)]',floor(1*maxChans));
    %     Idxs = unique(cell2mat(IdxsCell));
    %     Idxs = setdiff(Idxs, badIdxs);
end
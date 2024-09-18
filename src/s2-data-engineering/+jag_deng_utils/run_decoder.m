function [xhat] = run_decoder(TRAIN, TestX, TestZ,Idxs,method, varargin)
% inputs
%   TRAIN: structure of filter coefficients
%   TestX: nDOF x n samples of kinematic data
%   TestZ: 720 x n samples of feature data
%   Idxs: integer vector Selected feature indices
%   method: string decode method 'Standard' or 'DWPRR'
%   optional:
%       gains: nDOF x 2 matrix of gains, default 1
%       thresh: nDOF x 2 matrix of thresholds, default 0.2
% outputs:
%   xhat: nDOF x n samples of decode output
% smw 3/2017
%
% based on: C:\Users\Marta Iversen\Downloads\Scripts\runDecode_jag.m

KalmanGain = ones(size(TestX, 1), 2);
KalmanThresh = 0.2*ones(size(TestX, 1), 2);

if nargin > 5
    KalmanGain = varargin{1};
end
if nargin > 6
    KalmanThresh = varargin{2};
end

xhat = zeros(size(TestX));
xhatSS = xhat;
for j = 1: size(TestX,2)
    switch method
        case 'standard'
            xhat(:,j) = jag_deng_utils.test_kalman(TestZ(Idxs,j),TRAIN,[-1./KalmanGain(:,2),1./KalmanGain(:,1)],j==1);
        case 'DWPRR'
            tempZ = TestZ(Idxs,j)-minZ;
            %             tempZ = nthroot(tempZ,3);
            tempZ = tempZ./normalizerZ;%max(TestZ(Idxs,:),[],2);
            tempZ = nthroot(tempZ,3);
            tempZAll(:,j) = tempZ;
            tempZ = [tempZ; 1];
            tempFeat = (w'*tempZ).^3;% previously tempFeat(:,j) but crashed on j = 2
            
            tempFeatAll(:,j) = tempFeat;
            xhat(:,j) = jag_deng_utils.test_kalman(tempFeat,TRAIN,[-1./KalmanGain(:,2),1./KalmanGain(:,1)],j==1);
        case 'KalmanSS'
            error('DEPRECATED')
            % Ask T. Davis for file
            % xhat(:,j) = imprtKalman_testSS(TestZ(Idxs,j),TRAIN,[-1./KalmanGain(:,2),1./KalmanGain(:,1)],j==1);
    end
    
    % threshold output
    pos = (xhat(:,j)>=0);
    if any(pos)
        xhat(pos,j) = (xhat(pos,j).*KalmanGain(pos,1)-KalmanThresh(pos,1))./(1-KalmanThresh(pos,1)); %apply flexion gains/thresholds
    end
    if any(~pos)
        xhat(~pos,j) = (xhat(~pos,j).*KalmanGain(~pos,2)+KalmanThresh(~pos,2))./(1-KalmanThresh(~pos,2)); %apply extension gains/thresholds
    end
    
    xhat(xhat(:,j)<0 & pos, j) = 0;
    xhat(xhat(:,j)>0 & ~pos, j) = 0;
    
    xhat(xhat(:,j)<-1, j) = -1;
    xhat(xhat(:,j)>1, j) = 1;
    
end
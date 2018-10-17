classdef L1Loss < dagnn.Loss
%EPE
  methods
    function outputs = forward(obj, inputs, params)
      [w,h,~,~] = size(inputs{1});
%       if w~=size(inputs{2},1); inputs{2} = imresize(inputs{2}, [w h]);end
      %0.5*(c-x)^2
      t = bsxfun(@minus,inputs{2},inputs{1});
      t = gather(t);
      t = reshape(t,1,[]);
      t(abs(t)>1) = abs(t(abs(t)>1));
      t(abs(t)<1) = 0.5*(t(abs(t)<1)).^2;
      outputs{1} = sum(t)/(w*h*3); 
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + double(gather(outputs{1}))) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      [w,h,~,~] = size(inputs{2});
%       if w~=size(inputs{2},1); inputs{2} = imresize(inputs{2}, [w h]);end
      %x -y ;
      Y = gather(bsxfun(@minus,inputs{1},inputs{2}));
      Y(Y>1)= 1;  % x-y>1 
      Y(Y<-1) = -1; % y-x<1
      derInputs{1} = gpuArray(bsxfun(@times, derOutputs{1},Y));
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function obj = HuberLoss(varargin)
      obj.load(varargin) ;
    end
  end
end


function [net,stats] = cnn_train_dag_sp(net, imdb,imageName, getBatch,varargin)
% ver2: Solver is added

%CNN_TRAIN_DAG Demonstrates training a CNN using the DagNN wrapper
%    CNN_TRAIN_DAG() is similar to CNN_TRAIN(), but works with
%    the DagNN wrapper instead of the SimpleNN wrapper.

% Copyright (C) 2014-16 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
addpath(fullfile(vl_rootnn, 'examples'));

opts.expDir = fullfile('data','exp') ;
opts.continue = false ;
opts.batchSize = 256 ;
opts.numSubBatches = 1;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [] ;
opts.prefetch = false ;
opts.numEpochs = 2 ;
opts.learningRate = 0.001 ;
opts.weightDecay = 0.0005 ;
opts.solver = @solver.sgd; % Empty array - optimised SGD solver

[opts, varargin] = vl_argparse(opts, varargin);
if isempty(opts.solver)
  opts.solverOpts.momentum = 0.9;
else
  assert(isa(opts.solver, 'function_handle') && nargout(opts.solver) == 2,...
    'Invalid solver - a function handle with two outputs expected.');
  % A call without any input arg - def opts
  opts.solverOpts = opts.solver();
end

opts.saveSolverState = true ;
opts.randomSeed = 0 ;
opts.profile = false ;
opts.parameterServer.method = 'mmap' ;
opts.parameterServer.prefix = 'mcn' ;
opts.derOutputsG = {'L1loss', 1};
opts.derOutputsD = {'softmaxlog', 1};

opts.extractStatsFn = @extractStats ;
opts.plotStatistics = false;
opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end
if isnan(opts.val), opts.val = [] ; end

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------

evaluateMode = isempty(opts.train) ;
if ~evaluateMode
  if isempty(opts.derOutputsG) || isempty(opts.derOutputsD)
    error('DEROUTPUTS must be specified when training.\n') ;
  end
end


storeDir = 'output/';
% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

modelPath = @(ep) fullfile(storeDir, sprintf('net-epoch-%d.mat', ep));
modelFigPath = fullfile(storeDir, 'net-train.pdf') ;

%重新加载模型参数
start = opts.continue * findLastCheckpoint(opts.expDir) ;
if start >= 1
  fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
  [net, state, stats] = loadState(modelPath(start)) ;
  
else
  state = [];
end




for epoch=1:1:2

  % Set the random seed based on the epoch and opts.randomSeed.
  % This is important for reproducibility, including when training
  % is restarted from a checkpoint.

  rng(epoch + opts.randomSeed) ;
  %prepareGPUs(opts, epoch == start+1) ;

  %bae
  %imdb.images.noise = single(randn(1, 1, 100, imdb.nSample, 'single'));
  %imdb.images.label0 = -single(ones(1, 1, 1, imdb.nSample, 'single')+imdb.label_std*randn(1, 1, 1, imdb.nSample, 'single'));
  %imdb.images.label1 = single( ones(1, 1, 1, imdb.nSample, 'single')+imdb.label_std*randn(1, 1, 1, imdb.nSample, 'single')); 
  
  % Train for one epoch.
  params = opts ;
  params.epoch = epoch ;
  params.learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;
  params.train = opts.train(randperm(numel(opts.train))) ; % shuffle
  params.val = opts.val(randperm(numel(opts.val))) ;
  params.imdb = imdb ;
  params.getBatch = getBatch;

  if numel(opts.gpus) <= 1
    [net, state] = processEpoch(net, state, params, 'train',imageName,storeDir,epoch) ;
    %[net, state] = processEpoch(net, state, params, 'val',imageName,storeDir,epoch) ;
    %disp(state)
    if ~evaluateMode
      %saveState(modelPath(epoch), net, state) ;
      if epoch>5 
          delete(modelPath(epoch-5)); 
      end
    end
    lastStats = state.stats ;
  else
    spmd
      [net, state] = processEpoch(net, state, params, 'train',imageName,storeDir ,epoch) ;
      %[net, state] = processEpoch(net, state, params, 'val',imageName,storeDir,epoch) ;
      if labindex == 1 && ~evaluateMode
        %saveState(modelPath(epoch), net, state) ;
        if epoch>5 
          delete(modelPath(epoch-5)); 
        end
      end
      lastStats = state.stats ;
    end
    lastStats = accumulateStats(lastStats) ;
  end

  stats.train(epoch) = lastStats.train ;
%  stats.val(epoch) = lastStats.val ;
  clear lastStats ;
  %saveStats(modelPath(epoch), stats) ;

  if opts.plotStatistics
    switchFigure(1) ; clf ;
    plots = setdiff(...
      cat(2,...
      fieldnames(stats.train)', ...
      fieldnames(stats.val)'), {'num', 'time'}) ;
    for p = plots
      p = char(p) ;
      values = zeros(0, epoch) ;
      leg = {} ;
      for f = {'train', 'val'}
        f = char(f) ;
        if isfield(stats.(f), p)
          tmp = [stats.(f).(p)] ;
          values(end+1,:) = tmp(1,:)' ;
          leg{end+1} = f ;
        end
      end
      subplot(1,numel(plots),find(strcmp(p,plots))) ;
      plot(1:epoch, values','o-') ;
      xlabel('epoch') ;
      title(p) ;
      legend(leg{:}) ;
      grid on ;
    end
    drawnow ;
    print(1, modelFigPath, '-dpdf') ;
  end
end

% With multiple GPUs, return one copy
if isa(net, 'Composite'), net = net{1} ; end

% -------------------------------------------------------------------------
function [net, state] = processEpoch(net, state, params, mode, imageName,storeDir,epoch2)
% -------------------------------------------------------------------------
% Note that net is not strictly needed as an output argument as net
% is a handle class. However, this fixes some aliasing issue in the
% spmd caller.
netG = net(1);
netD = net(2);

% initialize with momentum 0
if isempty(state)
  stateG.solverState = cell(1, numel(netG.params)) ;
  stateD.solverState = cell(1, numel(netD.params)) ;
  state = [stateG, stateD];
end

stateG = state(1);
stateD = state(2);

% move CNN  to GPU as needed
numGpus = numel(params.gpus) ;
if numGpus >= 1
  netG.move('gpu') ;
  netD.move('gpu') ;
  
%   state.momentum = cellfun(@gpuArray, state.momentum, 'uniformoutput', false) ;
end
if numGpus > 1
  parserv = ParameterServer(params.parameterServer) ;
  netG.setParameterServer(parserv) ;
  
  parserv = ParameterServer(params.parameterServer) ;
  netD.setParameterServer(parserv) ;
  
else
  parserv = [] ;
end

% profile
if params.profile
  if numGpus <= 1
    profile clear ;
    profile on ;
  else
    mpiprofile reset ;
    mpiprofile on ;
  end
end

num = 0 ;
epoch = params.epoch ;
subset = params.(mode);
im_idx = randi(length(subset));subset = repmat(subset(im_idx),1,100);
% subset = randi(length(subset),1,1000);
adjustTime = 0 ;

stats.num = 0 ; % return something even if subset = []
stats.time = 0 ;

start = tic ;

maxBatchNumber = 100;
%batch 1     nume1(subset) 100
for t=1:params.batchSize:maxBatchNumber
    if t == 100
        fprintf('%s: epoch %02d: %3d/%3d:', mode, epoch, ...
            fix((t-1)/params.batchSize)+1, ceil(numel(subset)/params.batchSize)) ;
    end
    batchSize = min(params.batchSize, numel(subset) - t + 1) ;
    %batchStart = t + (labindex-1);
    %batchEnd = min(t+params.batchSize-1, numel(subset)) ;
    
    %batch = subset(batchStart : numlabs : batchEnd);
    %inputs = params.getBatch(params.imdb, batch) ; %bae
    
    %noise = inputs{4};
    %data = inputs{4};
    %label0= inputs{6};
    %label1= inputs{8};
    
    %train input%
    im = imread(strcat('testInput\',imageName)) ;
    im_ = single(im)/255 ; % note: 0-255 range
    im_ = imresize(im_, netD.meta.normalization.imageSize(1:2)) ;
    noise = gpuArray(im_);
    data = noise;
    
    Iguided = imguidedfilter(im_);
    Idetail = (im_ - Iguided)*1.5;
    detail = gpuArray(Idetail);
%     figure(2);imshow((Idetail*1.5)+Iguided);
        
    if strcmp(mode, 'train')
        % Update G with L1
        netG.mode = 'normal' ;
        netG.forward({'data',noise});
        netG.backward(params.derOutputsG);
        stateG = accumulateGradients(netG, stateG, params, batchSize, parserv) ;
        
        netG.forward({'data',noise});
        label_est =  netG.vars(end-1).value; %reconstructed image
        % update D
%         netD.mode = 'normal' ;
%         netD.forward({'data', label_est, 'label', label0});
%         netD.backward( params.derOutputsD);
%         stateD = accumulateGradients(netD, stateD, params, batchSize, parserv) ;
% 
%         netD.forward({'data', data, 'label', label1});
%         netD.backward( params.derOutputsD);
%         stateD = accumulateGradients(netD, stateD, params, batchSize, parserv) ;

        
        % update G by backpropagated grad in D
        netG.mode = 'normal'; 
        netD.mode = 'normal';
        %label1 = label1.*0+1;
%         im_ = single(label_est) ; % note: 0-255 range
%         im_ = imresize(im_, netD.meta.normalization.imageSize(1:2)) ;
%         im_ = bsxfun(@minus, im_, netD.meta.normalization.averageImage) ;
        netD.forward({'data',data*255}); %get ground truth estimation of black box
        scores = netD.vars(netD.getVarIndex('prob')).value ;
        scores = squeeze(gather(scores)) ;
        [bestScore, best] = max(scores) ;
        [secondScore, random_pre] = max(scores(scores<max(scores)));
%         [worstScore, random_pre] = min(scores) ;

%         random_pre = randi([1, length(scores)]);
%         while random_pre == best
%             random_pre = randi([1, length(scores)]); %get a random prediction as ground truth
%         end
        cpu_est = gather(label_est); %put gpu reconstructed image to cpu
        est_guided = imguidedfilter(cpu_est) + detail;
        label_est = gpuArray(est_guided);
        netD.forward({'data',(label_est)*255,'label',random_pre});
        scores = netD.vars(netD.getVarIndex('prob')).value ;
        scores = squeeze(gather(scores)) ;
        [bestScore2, best2] = max(scores) ;
        netD.backward( params.derOutputsD); 
        netG.backward({params.derOutputsG{1}, netD.vars(1).der});
        
        figure(1) ; clf ; imshow([data label_est]) ;
        title(sprintf('GT: %s (%d), score %.3f. Pred: %s (%d), score %.3f ',...
        netD.meta.classes.description{best}, best, bestScore, netD.meta.classes.description{best2}, best2, bestScore2)) ;drawnow;
    
        stateG = accumulateGradients(netG, stateG, params, batchSize, parserv) ;
        
        if epoch2 == 2 & t == maxBatchNumber %bae
%             im_out = uint8((gather(label_est)+1)/2*255);
            im_out = uint8(gather(label_est)*255);    
            if params.batchSize >1 
                im_out = get_tensor2img(im_out); 
            end
            if ~exist(storeDir, 'dir'), mkdir(storeDir) ;end
            r2 = getGlobaly();
            imwrite(im_out, sprintf('%s/img_%.5d_RC%d_FC%d.png',storeDir, r2, best, best2));
             
            if best == best2
                r = getGlobalx();
                setGlobalx(r+1)
            end
        end
        
        

    else
        % validation
%         netG.mode = 'test' ;
%         netG.forward({'data',noise});
%         label_est =  netG.vars(end-1).value;
%         
%         netD.mode = 'test' ;
%         netD.eval({'data',cat(4, data, label_est) 'label', cat(4, label1, label0)});
    end
    
    
    % Get statistics.
    time = toc(start) + adjustTime ;
    batchTime = time - stats.time ;
    stats.num = num ;
    stats.time = time ;
    
    stats = params.extractStatsFn(stats,netD);
    stats = params.extractStatsFn(stats,netG);
    
    currentSpeed = batchSize / batchTime ;
    averageSpeed = (t + batchSize - 1) / time ;
    if t == 3*params.batchSize + 1
        % compensate for the first three iterations, which are outliers
        adjustTime = 4*batchTime - time ;
        stats.time = time + adjustTime ;
    end
    if t == 100
        fprintf(' %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;
        for f = setdiff(fieldnames(stats)', {'num', 'time'})
            f = char(f) ;
            fprintf(' %s: %.10f', f, stats.(f)) ;
        end
        fprintf('\n') ;
    end
end

% Save back to state.
stateG.stats.(mode) = stats ;
stateD.stats.(mode) = stats ;

if params.profile
  if numGpus <= 1
    stateG.prof.(mode) = profile('info') ;
    stateD.prof.(mode) = profile('info') ;
    profile off ;
  else
    stateG.prof.(mode) = mpiprofile('info');
    stateD.prof.(mode) = mpiprofile('info');
    mpiprofile off ;
  end
end
if ~params.saveSolverState
  stateG.solverState = [] ;
  stateD.solverState = [] ;
else
  stateG.solverState = cellfun(@gather, stateG.solverState, 'uniformoutput', false) ;
  stateD.solverState = cellfun(@gather, stateD.solverState, 'uniformoutput', false) ;
end


netG.reset() ;
netG.move('cpu') ;

netD.reset() ;
netD.move('cpu') ;

state = [stateG stateD];
net = [netG netD];


% -------------------------------------------------------------------------
function state = accumulateGradients(net, state, params, batchSize, parserv)
% -------------------------------------------------------------------------
numGpus = numel(params.gpus) ;
otherGpus = setdiff(1:numGpus, labindex) ;

for p=1:numel(net.params)

  if ~isempty(parserv)
    parDer = parserv.pullWithIndex(p) ;
  else
    parDer = net.params(p).der ;
  end

  switch net.params(p).trainMethod

    case 'average' % mainly for batch normalization
      thisLR = net.params(p).learningRate ;
      net.params(p).value = vl_taccum(...
          1 - thisLR, net.params(p).value, ...
          (thisLR/batchSize/net.params(p).fanout),  parDer) ;

    case 'gradient'
      thisDecay = params.weightDecay * net.params(p).weightDecay ;
      thisLR = params.learningRate * net.params(p).learningRate ;
      
      if isempty(params.solver)
        if isempty(state.solverState{p})
          state.solverState{p} = zeros(size(parDer), 'like', parDer);
        end
        
        state.solverState{p} = vl_taccum(...
          params.solverOpts.momentum,  state.solverState{p}, ...
          - (1 / batchSize), parDer) ;
        net.params(p).value = vl_taccum(...
          (1 - thisLR * thisDecay / (1 - params.solverOpts.momentum)),  ...
          net.params(p).value, ...
          thisLR, state.solverState{p}) ;
      else
        grad = (1 / batchSize) * parDer + thisDecay * net.params(p).value;
        % call solver function to update weights
        [net.params(p).value, state.solverState{p}] = ...
          params.solver(net.params(p).value, state.solverState{p}, ...
          grad, params.solverOpts, thisLR) ;
      end

    otherwise
      error('Unknown training method ''%s'' for parameter ''%s''.', ...
        net.params(p).trainMethod, ...
        net.params(p).name) ;
  end
end

% -------------------------------------------------------------------------
function stats = accumulateStats(stats_)
% -------------------------------------------------------------------------

for s = {'train', 'val'}
  s = char(s) ;
  total = 0 ;

  % initialize stats stucture with same fields and same order as
  % stats_{1}
  stats__ = stats_{1} ;
  names = fieldnames(stats__.(s))' ;
  values = zeros(1, numel(names)) ;
  fields = cat(1, names, num2cell(values)) ;
  stats.(s) = struct(fields{:}) ;

  for g = 1:numel(stats_)
    stats__ = stats_{g} ;
    num__ = stats__.(s).num ;
    total = total + num__ ;

    for f = setdiff(fieldnames(stats__.(s))', 'num')
      f = char(f) ;
      stats.(s).(f) = stats.(s).(f) + stats__.(s).(f) * num__ ;

      if g == numel(stats_)
        stats.(s).(f) = stats.(s).(f) / total ;
      end
    end
  end
  stats.(s).num = total ;
end

% -------------------------------------------------------------------------
function stats = extractStats(stats, net)
% -------------------------------------------------------------------------
sel = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block})) ;
sel =[ sel, find(cellfun(@(x) isa(x,'dagnn.Pdist'), {net.layers.block}))] ;%bae
for i = 1:numel(sel)
  stats.(net.layers(sel(i)).outputs{1}) = net.layers(sel(i)).block.average ;
end

% -------------------------------------------------------------------------
function saveState(fileName, net_, state)
% -------------------------------------------------------------------------
netG = net_(1);
netD = net_(2);

netG = netG.saveobj() ;
netD = netD.saveobj() ;
net = [netG, netD];

save(fileName, 'net', 'state') ;

% -------------------------------------------------------------------------
function saveStats(fileName, stats)
% -------------------------------------------------------------------------
if exist(fileName)
  save(fileName, 'stats', '-append') ;
else
  save(fileName, 'stats') ;
end

% -------------------------------------------------------------------------
function [net, state, stats] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'net', 'state', 'stats') ;
netG = net(1);
netD = net(2);

netG = dagnn.DagNN.loadobj(netG) ;
netD = dagnn.DagNN.loadobj(netD) ;

net = [netG, netD];

if isempty(whos('stats'))
  error('Epoch ''%s'' was only partially saved. Delete this file and try again.', ...
        fileName) ;
end

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;

% -------------------------------------------------------------------------
function switchFigure(n)
% -------------------------------------------------------------------------
if get(0,'CurrentFigure') ~= n
  try
    set(0,'CurrentFigure',n) ;
  catch
    figure(n) ;
  end
end

% -------------------------------------------------------------------------
function clearMex()
% -------------------------------------------------------------------------
clear vl_tflow vl_imreadjpeg ;

% -------------------------------------------------------------------------
function prepareGPUs(opts, cold)
% -------------------------------------------------------------------------
numGpus = numel(opts.gpus) ;
if numGpus > 1
  % check parallel pool integrity as it could have timed out
  pool = gcp('nocreate') ;
  if ~isempty(pool) && pool.NumWorkers ~= numGpus
    delete(pool) ;
  end
  pool = gcp('nocreate') ;
  if isempty(pool)
    parpool('local', numGpus) ;
    cold = true ;
  end

end
if numGpus >= 1 && cold
  fprintf('%s: resetting GPU\n', mfilename)
  clearMex() ;
  if numGpus == 1
    opts.gpus
    gpuDevice(opts.gpus)
  else
    spmd
      clearMex() ;
      gpuDevice(opts.gpus(labindex))
    end
  end
end


function [im_out] = get_tensor2img(tensor_in)
f = size(tensor_in);
num_im_height = floor(sqrt(f(4)));
im_out = zeros(f(1).*num_im_height, f(2).*num_im_height, f(3), 'single');
im_out = uint8(im_out);

cnt = 0;
for y = 1:num_im_height
    for x = 1:num_im_height
        cnt = cnt + 1;
        idx_y = (y-1)*f(1)+1:y*f(1);
        idx_x = (x-1)*f(2)+1:x*f(2);
        im_out(idx_y,idx_x,:) = tensor_in(:,:,:,cnt);
        
    end
end

return;


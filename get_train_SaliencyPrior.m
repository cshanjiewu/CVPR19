function [ net ] = get_train_SaliencyPrior(varargin)
addpath('src');
opts.matconvnet_path = 'matconvnet-1.0-beta24/matlab/vl_setupnn.m';
% opts.contrib_path = 'matconvnet-1.0-beta24/matlab/vl_setupcontrib.m';

opts.idx_gpus = 1; % cpu/gpu settings 
%% 设置数据路径和大小
opts.imdb_path = ('list_Celeb_full.mat');
opts.imdb_img_size = [224 224];
opts.label_std = 0.2; %adding Gaussian noise to labels in training D

opts = vl_argparse(opts, varargin) ;
run(opts.matconvnet_path) ;
% run(opts.contrib_path) ;

%% imdb settings 
imdb = load(opts.imdb_path);
imdb = imdb.imdb;
imdb.label_std =opts.label_std;
imdb.nSample = numel(imdb.images.set);

imdb.gpus =opts.idx_gpus>0;
imdb.args  = {'NumThreads', 6, ...
            'Pack', ...
            'Interpolation', 'bicubic', ...
            'Resize', opts.imdb_img_size, ...
            'CropAnisotropy', [1, 1], ...
            'CropLocation', 'center', ...
            'CropSize', [1 1], ...
                };

% imdb.args{end+1} = 'Gpu'; 
% imdb.args{end+1} = 'Flip'; 


%% net settings 
opts.train.continue = false;
opts.train.solver = @solver.adam; % Empty array - optimised by SGD solver
opts.train.expDir = fullfile(mfilename) ;
opts.train.learningRate = 0.0002*ones(1,5);
opts.train.numEpochs = numel(opts.train.learningRate); 
opts.train.batchSize = 1;
opts.train.weightDecay = 0;
opts.train.derOutputsG = {'L1loss', 0.0001};
opts.train.derOutputsD = {'Dloss', 1000};
opts.train.gpus = opts.idx_gpus;
if opts.train.gpus == 0
    opts.train.gpus = [];
end

opts = vl_argparse(opts, varargin) ;

%% group x1
%% G layer
netG = dagnn.DagNN();
i = 0;

% 64x64 to 32x32·
conv_param = struct('f', [4 4 3 64], 'pad',  1, 'stride', 2, 'bias', false);
relu_param = 0.2;
i = i+1;  netG = get_Conv_dag(netG, i, 'data', sprintf('x%d',i), conv_param);
i = i+1;  netG = get_ReLU_dag(netG, i, sprintf('x%d',i-1), sprintf('x%d',i), relu_param);

% 32x32 to 16x16
conv_param = struct('f', [4 4 64 64*2], 'pad',  1, 'stride', 2, 'bias', false);
bn_param = struct('depth', 64*2, 'epsilon', 1e-5);
[netG, i] = get_Block_CBR( netG, i, conv_param, bn_param,relu_param); % 1

% 16x16 to 8x8
conv_param = struct('f', [4 4 64*2 64*4], 'pad',  1, 'stride', 2, 'bias', false);
bn_param = struct('depth', 64*4, 'epsilon', 1e-5);
[netG, i] = get_Block_CBR( netG, i, conv_param, bn_param,relu_param); % 1

% 8x8 to 4x4
conv_param = struct('f', [4 4 64*4 64*8], 'pad',  1, 'stride', 2, 'bias', false);
bn_param = struct('depth', 64*8, 'epsilon', 1e-5);
[netG, i] = get_Block_CBR( netG, i, conv_param, bn_param,relu_param); % 1

% 4x4 to 8x8
conv_param = struct('f', [4 4 64*8 64*4], 'pad',  1, 'stride', 2, 'bias', false);
bn_param = struct('depth', 64*4, 'epsilon', 1e-5);
i = i+1;  netG = get_DecConv_dag(netG, i, sprintf('x%d',i-1), sprintf('x%d',i), conv_param);
i = i+1;  netG = get_BN_dag(netG, i, sprintf('x%d',i-1), sprintf('x%d',i), bn_param);
i = i+1;  netG = get_ReLU_dag(netG, i, sprintf('x%d',i-1), sprintf('x%d',i), relu_param);

% 8x8 to 16x16
conv_param = struct('f', [4 4 64*4 64*2], 'pad',  1, 'stride', 2, 'bias', false);
bn_param = struct('depth', 64*2, 'epsilon', 1e-5);
i = i+1;  netG = get_DecConv_dag(netG, i, sprintf('x%d',i-1), sprintf('x%d',i), conv_param);
i = i+1;  netG = get_BN_dag(netG, i, sprintf('x%d',i-1), sprintf('x%d',i), bn_param);
i = i+1;  netG = get_ReLU_dag(netG, i, sprintf('x%d',i-1), sprintf('x%d',i), relu_param);

% 16x16 to 32x32
conv_param = struct('f', [4 4 64*2 64], 'pad',  1, 'stride', 2, 'bias', false);
bn_param = struct('depth', 64, 'epsilon', 1e-5);
i = i+1;  netG = get_DecConv_dag(netG, i, sprintf('x%d',i-1), sprintf('x%d',i), conv_param);
i = i+1;  netG = get_BN_dag(netG, i, sprintf('x%d',i-1), sprintf('x%d',i), bn_param);
i = i+1;  netG = get_ReLU_dag(netG, i, sprintf('x%d',i-1), sprintf('x%d',i), relu_param);

% 32x32 to 64x64
conv_param = struct('f', [4 4 64 3], 'pad',  1, 'stride', 2, 'bias', false);
i = i+1;  netG = get_DecConv_dag(netG, i, sprintf('x%d',i-1), sprintf('x%d',i), conv_param);
i = i+1;  
netG.addLayer('Tanh', dagnn.Tanh(), sprintf('x%d',i-1), 'Tanh');
netG.addLayer('L1loss', L1Loss(), {sprintf('x%d',i-1), 'data'}, 'L1loss') ;

netG.initParams();
netG.meta.trainOpts = opts.train;

%% D layer
% netD = dagnn.DagNN();
% i = 0;
% 
% % 64x64 to 32x32
% conv_param = struct('f', [4 4 3 64], 'pad',  1, 'stride', 2, 'bias', false);
% relu_param = 0.2;
% i = i+1;  netD = get_Conv_dag(netD, i, 'data', sprintf('x%d',i), conv_param);
% i = i+1;  netD = get_ReLU_dag(netD, i, sprintf('x%d',i-1), sprintf('x%d',i), relu_param);
% 
% % 32x32 to 16x16
% conv_param = struct('f', [4 4 64 64*2], 'pad',  1, 'stride', 2, 'bias', false);
% bn_param = struct('depth', 64*2, 'epsilon', 1e-5);
% [netD, i] = get_Block_CBR( netD, i, conv_param, bn_param,relu_param); % 1
% 
% % 16x16 to 8x8
% conv_param = struct('f', [4 4 64*2 64*4], 'pad',  1, 'stride', 2, 'bias', false);
% bn_param = struct('depth', 64*4, 'epsilon', 1e-5);
% [netD, i] = get_Block_CBR( netD, i, conv_param, bn_param,relu_param); % 1
% 
% % 8x8 to 4x4
% conv_param = struct('f', [4 4 64*4 64*8], 'pad',  1, 'stride', 2, 'bias', false);
% bn_param = struct('depth', 64*8, 'epsilon', 1e-5);
% [netD, i] = get_Block_CBR( netD, i, conv_param, bn_param,relu_param); % 1
% 
% % 4x4 to 1x1
% conv_param = struct('f', [4 4 64*8 1], 'pad',  0, 'stride', 2, 'bias', false);
% i = i+1;  netD = get_Conv_dag(netD, i, sprintf('x%d',i-1), sprintf('x%d',i), conv_param);
% i = i+1;  netD.addLayer('D_loss', dagnn.Loss('loss', 'logistic'), { sprintf('x%d',i-1),'label'},'logistic');
%load the pretrained model
netD = dagnn.DagNN.loadobj(load('net/imagenet-googlenet-dag')) ;
netD.addLayer('Dloss', dagnn.Loss('loss', 'softmaxlog'), {'prob', 'label'}, 'Dloss');
netD.addLayer('error', dagnn.Loss('loss', 'classerror'), {'prob','label'}, 'error') ;
% netD.initParams();
opts.trainD = opts.train; 
opts.trainD.learningRate = 0;
netD.meta.trainOpts = opts.trainD;

net = [netG, netD];
% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------
imageName = dir(fullfile('D:\ILSVRC2012_img_val\'));
%D:\ILSVRC2012_img_val\
global x ;
global y ;
setGlobalx(0);
setGlobaly(0);

prepareGPUs(opts, true) ;

for i=3:1:numel(imageName)
    cnn_train_dag_sp(net, imdb,imageName(i).name, @getBatchHdd, opts.train, ...
                                       'val', find(imdb.images.set == 2)) ;
    r = getGlobaly;
    setGlobaly(r+1);
    
    fenzi = getGlobalx();
    fenmu = getGlobaly();
    fprintf('Failue rate: %d-----------------------------------------------\n', fenzi/fenmu);
           
end

%[net, info] = cnn_train_dag_sp(net, imdb, @getBatchHdd, opts.train, ...
%                                       'val', find(imdb.images.set == 2),ImageName) ;
                     
return;


function inputs = getBatchHdd(imdb, batch)
% --------------------------------------------------------------------
args_data = [{imdb.images.data(batch)} imdb.args];

noise = imdb.images.noise(:,:,:, batch);
label0 = imdb.images.label0(:,:,:, batch);
label1 = imdb.images.label1(:,:,:, batch);

data = vl_imreadjpeg(args_data{:});
data = data{1}/255*2-1; % set in the range of [-1:1]

if (imdb.gpus == true)
    noise = gpuArray(noise);
    data = gpuArray(data);
    label0 = gpuArray(label0);
    label1 = gpuArray(label1);    
end

inputs = ({'noise', noise, 'data', data, 'label0',label0 'label1',label1} );

return;



% -------------------------------------------------------------------------
function clearMex()
% -------------------------------------------------------------------------
clear vl_tflow vl_imreadjpeg ;


% -------------------------------------------------------------------------
function prepareGPUs(opts, cold)
% -------------------------------------------------------------------------
numGpus = 1 ;
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
    temp = 1;
    gpuDevice(temp)
  else
    spmd
      clearMex() ;
      gpuDevice(opts.gpus(labindex))
    end
  end
end





%% Prerequisite: 
%% 1. Download CelebA database at: https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing
%% 2. Download Pretrained DCGAN at: https://www.dropbox.com/s/fvasd83oqgse7xr/net_dcgan.mat?dl=0
%% 将路径加入当前环境
addpath('src');
addpath('net');
%% 设置使用gpu还是cpu
opts.idx_gpus = 1; % 0: cpu 1: gpu

%% 1) Install MatConvNet
opts.install.cuda_path = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v10.0';  %设置你的cuda文件夹
opts.install.matconvnet_path = 'matconvnet-1.0-beta24/matlab/vl_setupnn.m';
% untar('http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta25.tar.gz') ;
%% 安装matconvnet库
run(opts.install.matconvnet_path) ;
%% 编译matconvnet
if opts.idx_gpus > 0
    vl_compilenn('enableGpu', true,'cudaRoot', opts.install.cuda_path, 'cudaMethod', 'nvcc') ;
else 
    vl_compilenn('enableGpu', false);
end    

fprintf('Install MatConvNet.... Done \n')
%% 2) Copy customized layers
src = 'src/matlab/*';
dst = 'matconvnet-1.0-beta24/matlab/';
copyfile(src, dst);

src = 'src/+solver/*';
dst = 'matconvnet-1.0-beta24/examples/+solver';
copyfile(src, dst);

fprintf('Copy customized layers.... Done \n')
%% 3) Get imdb (CelebA)
% 
% opts.imdb.im_in_dir  = 'data/img_align_celeba'; % CelebA data should be ready prior to the beginning of this script 
% opts.imdb.im_out_dir = 'data/img_align_celeba_crop';
% 
% get_img2re_crop_img(opts.imdb);
% 
% opts.imdb.im_in_dir  = opts.imdb.im_out_dir;
% opts.imdb = rmfield(opts.imdb,'im_out_dir');
% 
% get_img2list(opts.imdb);
% 
% fprintf('Get imdb.... Done \n')
%% 4) Test dcgan with a pre-trained model
% opts.test.save_img_path = 'test_img_sp'; % path to the output images 
% opts.test.num_images = 32; % num of images to be generated
% opts.test.idx_gpus = opts.idx_gpus;
% opts.test.net_path = 'get_train_SaliencyPrior/net-epoch-10.mat'; 
% opts.test.imdb_path = ('list_Celeb_full.mat');
% 
% get_test_SaliencyPrior(opts.test);
% 
% fprintf('Test dcgan with a pre-trained model.... Done \n')
%% 5) Train dcgan
% opts.train.matconvnet_path = 'matconvnet-1.0-beta25/matlab/vl_setupnn.m';
% opts.train.idx_gpus = opts.idx_gpus;
% 
% get_train_DCGAN(opts.train);
% 
% fprintf('Train dcgan.... Done \n')

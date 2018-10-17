function vl_setupcontrib()

root = vl_rootnn() ;
addpath(fullfile(root, 'contrib')) ;
vl_contrib('setup', 'autonn') ;
vl_contrib('setup', 'mcnExtraLayers') ;
vl_contrib('setup', 'mcnRobustLoss') ;
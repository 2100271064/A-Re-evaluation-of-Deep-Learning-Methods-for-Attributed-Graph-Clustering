clc;
clear;
addpath('./libsvm-3.11/matlab/');
addpath('./svminfor/')
addpath('./code_coregspectral')
dataset_name = 'wiki';
path = sprintf('./data/myData/new/%s.mat', dataset_name);
load(path); % 数据集：fea（特征）、gnd（标签）、W（邻接矩阵）


if strcmpi(dataset_name,'pubmed') == 0
    components = 500;
    if strcmpi(dataset_name,'dblp') 
        components = 256;
    end
    % 应用 PCA
    [coeff, score, latent] = pca(full(fea), 'NumComponents', components);
    % 获取降维后的特征矩阵
    fea = fea * coeff;
end

n_outliers = 0;
if strcmpi(dataset_name,'citeseer') 
    n_outliers = 1;
elseif strcmpi(dataset_name,'cora')
    n_outliers = 7;
elseif strcmpi(dataset_name,'wiki')
    n_outliers = 5;
end

process_gnd = gnd(1:end-n_outliers,:); % 去除n_outliers点

num_views=2;
numClust=length(unique(process_gnd)); 
noise = 0.4;    % corruption level, 0.4 is good for cora
layers = 3;     % number of layers to stack, 3 for cora 
projev = 1.5;

fprintf('======================================\n');
disp('create A_n');
%create A_bar
n = size(fea,1);
A_bar = W + speye(n); % 加入I
d = sum(A_bar);
d_sqrt = 1.0./sqrt(d);
d_sqrt(d_sqrt == Inf) = 0;
DH = diag(d_sqrt);
DH = sparse(DH);
A_n = DH * sparse(A_bar) * DH;   %d*d
disp('compute A_n finished');

gcn = A_n * fea;   %d*d * d*n
[n,m] = size(gcn);

[allhx] = mSDA(fea', noise, layers, A_n);
process_allhx = allhx(:,1:end-n_outliers); % 去除n_outliers点
Z0 = process_allhx(end-m+1:end,:);
Z1 = Z0' * Z0;
Z2 = (abs(Z1) + abs(Z1'))/2;

[V Eval F P R nmi_v avgent AR C,ACCb] = baseline_spectral_onRW(Z2,numClust,process_gnd,projev);
fprintf('Clustering Results on Z2: ACC=%f, nmi score=%f, F=%f, P=%f, R=%f, avgent=%f,  AR=%f, \n',ACCb(1), nmi_v(1),F(1),P(1),R(1),avgent(1),AR(1));
fprintf('======================================\n');
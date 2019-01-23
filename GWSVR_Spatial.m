function [GW_pre,GW_mse,GW_CV]=GWSVR(train_x,train_y,test_x,test_y,train_Cor,test_Cor) 
% This is a test program for GWSVR,
% Please install LIBSVM(verson 3.22) before using this program
% ----------------------Input information----------------------
% Training dataset: train_x [M1*N] and train_y [M1*1]
% Testing dataset: test_x [M2*N] and test_y [M2*1]
% Spatial coordinates: train_Cor [M1,2] and train_Cor [M2,2]
% M1 and M2 are the size of training and testing samples
% N is the size of inpute features or indenpent variables
% If test_y is unknown than a M2-Dimensional zero vector can be defiend
% ----------------------Output information---------------------
% GW_Pre [M2*1]: predicted value of test_y
% GW_mse [M2*1]: Mean square error at each testing sample
% GW_CV  [M2*1]: Minimum cross-validation value at each testing sample

% Normalization preprocessing 
[train_x,test_x] = scaleForSVM(train_x,test_x,0,1);
[train_y_scale,test_y_scale,ps] = scaleForSVM(train_y,test_y,0,1);

% Compute Distances (Euclidean)
l_1=length(test_y);
l_2=length(train_y);
DisM=zeros(l_1,l_2);
for i=1:l_1
    for j=1:l_2
        DisM(i,j)=sqrt((test_Cor(i,1)-train_Cor(j,1))^2+(test_Cor(i,2)-train_Cor(j,2))^2);
    end
end

% construct a local model for each testing point
h = waitbar(0,'Please wait...');
for i=1:l_1
     waitbar(i/l_1,h);
     % Define bandwidth h in geographically weighted function
     % ... and determinate a optimal weight  
     
     %----------------Form 1///----------------
     count=0;
     nn=median(DisM(i,:));
     for bandwidth=0.3*nn:0.3*nn:3*nn     % The parameters 0.3 can be modified
         for j=1:l_2
             weight(j)=exp(-(DisM(i,j)/bandwidth)^2);
%              if DisM(i,j)>=bandwidth
%                  weight(j)=0;
%              else
%                  weight(j)=(1-(DisM(i,j)/bandwidth)^2)^2;
%              end
         end
         weight=weight/sum(weight);
         [bestmse,bestc,bestg] = SVMcgForGWRegress(train_y_scale,train_x,weight);
         cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.01'];
         count=count+1;
         CV(count)=bestmse;
     end
     GW_CV(i,1:count)=CV(1:count);
     Ind=find(CV==min(CV));
     optimal_bandwidth=0.3*nn*Ind(1); %  Parameters 0.3 should be consistent with that in Line 40
     for j=1:l_2
         weight(j)=exp(-(DisM(i,j)/optimal_bandwidth)^2);
%          if DisM(i,j)>=optimal_bandwidth
%              weight(j)=0;
%          else
%              weight(j)=(1-(DisM(i,j)/optimal_bandwidth)^2)^2;
%          end
     end
     weight=weight/sum(weight);
     %----------------///Form 1----------------

     %----------------Form 2///----------------
%      count=0;
%      Dis_Sort=sort(DisM(i,:));   
%      for nn=25:50       % This parameters 25 and 50 can be modified 
%          for j=1:l_2
%              if DisM(i,j)>Dis_Sort(nn)   
%                  weight(j)=0;
%              else
%                  weight(j)=(1-(DisM(i,j)/Dis_Sort(nn))^2)^2;
%              end
%          end
%          [bestmse,bestc,bestg] = SVMcgForGWRegress(train_y_scale,train_x,weight);
%          cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.01'];
%          count=count+1;
%          CV(count)=bestmse;
%          clc
%      end
%      GW_CV(i,1:count)=CV(1:count);
%      Ind=find(CV==min(CV));
%      Optimal_bandwidth=Dis_Sort(24+Ind(1));
%      for j=1:l_2
%          if DisM(i,j)>Optimal_bandwidth
%              weight(j)=0;
%          else
%              weight(j)=(1-(DisM(i,j)/Optimal_bandwidth)^2)^2;
%          end
%      end
     %----------------///Form 2:----------------
     
    
%   Model train for each testing point
    cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg),' -s 3 -p 0.01'];
    model = svmtrain(weight',train_y_scale, train_x,cmd);
    
    [ptrain, GW_mse] = svmpredict(train_y_scale, train_x, model);
    ptrain = mapminmax('reverse',ptrain',ps);
    ptrain = ptrain';
    GW_fit=ptrain;
    
%  Predict the output value of test_x
    [ptest, test_mse,desicion] = svmpredict(test_y_scale(i,1),test_x(i,:), model);
    ptest = mapminmax('reverse',ptest',ps);
    GW_mse(i,1)=test_mse(2);
    GW_pre(i,1)=ptest;
end
close(h);

% figure;
% plot(GW_pre,test_y,'o');
% grid on;
% legend('original','predict');
% title('Train Set Regression Predict by SVM');


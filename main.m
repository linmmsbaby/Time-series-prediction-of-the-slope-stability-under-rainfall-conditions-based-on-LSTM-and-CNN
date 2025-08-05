%% This is a MATLAB file that created by Mansheng Lin, Augest 5, 2025

% run this file by using MATLAB R2024b

% Paper name: Time series prediction of the slope stability under rainfall conditions based on LSTM and CNN

% Authors: Mansheng Lin, Yucheng Lu, Yan Li, Gongfa Chena, Bingxiang Yuan

%%




clear
path = '.\inp';
path2 = '.\slopefiles';

x = [-20,-10,0:1:5,10:10:100];

ypartnumber = 20; 
data = {};
FS = {};
xy = {};
iall = [];
Yall = {};
PORlist0 =[];

i = 2 ;

rainforcenow = [];
fs = [];
go = 0;

eend = 72;

yall = {};
PORthisslope = [];
for rainfalltime = 1:eend

    filename = [path,'\', 'slope',num2str(i),'_',num2str(rainfalltime),'odbPOR.txt'];           
    PORlist = readmatrix(filename);
    if max(size(PORlist))==1
        PORlist0 =[ PORlist0; i,rainfalltime];
    end
    y = [];
    xxx = [];
    PORthisslopetime = [];
    for xx = x
        idx = find(abs(PORlist(:,1)-xx)<2);
        PORlistsub = PORlist(idx,:);
        ymax = max(PORlistsub(:,2));  ymin = min(PORlistsub(:,2));
        y = [y;linspace(ymin,ymax,ypartnumber)']; xxx = [xxx;repmat(xx,[ypartnumber,1])];
        PORthisline = [];
        for yy = linspace(ymin,ymax,ypartnumber)
            dist = ((PORlist(:,1)-xx).^2+(PORlist(:,2)-yy).^2).^0.5;
            location = find(dist == min(dist));
            PORthispoint = PORlist(location(1),3);
            PORthisline = [PORthisline;PORthispoint ];
        end
        PORthisslopetime = [PORthisslopetime; PORthisline];
    end
    PORthisslope = [PORthisslope,PORthisslopetime];
    filename2 = [path2,'\', 'slope',num2str(i),'_',num2str(rainfalltime),'.txt'];
    sloped = readmatrix(filename2);
    rainforcenow = [rainforcenow;sloped(end,1)];           
end

iall = [iall;i];
xy{end+1} = [xxx,y];
data{end+1,1} = PORthisslope;

save data.mat
%% 

clear

load data.mat

load trainedLSTM.mat


idxConstantValue = {};
for i = 1:numel(data)
    idxConstantValue{end+1,1} = data{i}(idxConstant,:);
end

for i = 1:numel(data)
    data{i}(idxConstant,:) = [];
end


for n = 1:size(data,1)
    X = data{n};
    XTest{n} = (X(:,1:end-1) - muX) ./ sigmaX;
    TTest{n} = (X(:,2:end) - muT) ./ sigmaT;
end

YTest = predict(trainedNet,XTest,SequencePaddingDirection="left");


% 反归一
YTestR={};
for n = 1:size(YTest,1)
    Y = YTest{n};
    YTestR{n} = Y.*sigmaT+muT;
end

rmse = [];
for i = 1:length(YTestR)
    X = data{i};
    rmse(i) = sqrt(mean((YTestR{i} - X(:,2:end)).^2,"all"));
end
mean(rmse)

save LSTMpredictedResults.mat

%% getPredictedPORdistributions
clear

path = '.\inp';

load('LSTMpredictedResults.mat','YTestR','idxConstant','idxConstantValue','data','xy')

i = 2;

eend = 71;

E = [];
for ii = 1:eend
 
    y = zeros(size(idxConstant));
    y(idxConstant) = idxConstantValue{:}(:,ii+1);
    y(~idxConstant) = YTestR{:}(:,ii); 

    yT = zeros(size(idxConstant));
    yT(idxConstant) = idxConstantValue{:}(:,ii+1);
    yT(~idxConstant) = data{:}(:,ii);


    filename = [path,'\', 'slope',num2str(i),'_',num2str(ii+1),'odbPOR.txt']; 
    PORlist = readmatrix(filename);

    S = xy{:};
    F = scatteredInterpolant(S(:,1),S(:,2),y);
    F.Method = 'natural';
    PORlistnew = F(PORlist(:,1),PORlist(:,2));
    Ematlab = (sum((PORlist(:,3)-PORlistnew).^2)/length(PORlistnew)).^0.5;

    E = [E; Ematlab];

    PORlistnew = [PORlist(:,1:2), PORlistnew]; 
    
    file = fopen([path,'\', 'slope',num2str(i),'_',num2str(ii+1),'odbPORnew.txt'],'w+');% ,'n',"US-ASCII"
    for j = 1:size(PORlistnew,1)
        layerPropsub = string(PORlistnew(j,:));
        layerPropsub(:,1:end-1) = layerPropsub(1:end-1)+',';
        layerPropsub = join(layerPropsub);
        fprintf(file, '%0s\r\n', layerPropsub);
    end
    fclose(file);
end

%% 
clear
pathslope =  '.\slopefiles';
pathodb = '.\inp';

i = 2;
eend = 72;
slope = []; % 4-D matrix input
for rainfalltime = 1:eend
    txtslope = [pathslope,'\', 'slope',num2str(i),'_',num2str(rainfalltime),'.txt'];
    txtslopepor = [pathodb,'\', 'slope',num2str(i),'_',num2str(rainfalltime),'odbPORnew.txt'];
    if ~exist(txtslopepor)
        txtslopepor = [pathodb,'\', 'slope',num2str(i),'_',num2str(rainfalltime),'odbPOR.txt'];
    end 
    [grid_slope,~,~] = genSlopeMatrix(txtslope,txtslopepor);
    slope = cat(4,slope,grid_slope); % input   
end   

save('slopedata.mat')
%% 
clear
load('slopedata.mat')
load('calFS.mat')
load('trainedCNN.mat') 
load('datarescle.mat') 

for i = 1:4
    mu = datarescle(i,1);
    sigma = datarescle(i,2);
    slope(:,:,i,:) = (slope(:,:,i,:) - mu) ./ sigma;
end


YPredicted = predict(trainedNet,slope); 

if iscell(YPredicted)
    YPredicted = cell2mat(YPredicted);
    activationYValidation = cell2mat(activationYValidation);
end


YValidationD = FOS;
YPredictedD = YPredicted;


rmse = sqrt(mean((YPredictedD - YValidationD).^2))
error = YPredictedD - YValidationD;
mean(abs(error))

R = corrcoef(YValidationD, YPredictedD);
R2 = R(1,2)^2;

figure
left_color = [0 0 0];
right_color = [222,142,105]./255;
set(gcf,'defaultAxesColorOrder',[left_color; right_color]);
yyaxis right
bar(abs(error),'FaceColor',right_color,'EdgeColor',right_color)
title(['RMSE = ' , num2str(rmse),', R^2 = ',num2str(R2) ])
ylim([0,0.8])
ylabel("Error")
xlabel("Sample no.")
yyaxis left
plot(YPredictedD,LineWidth=1,Color='k',LineStyle='-',Marker='square',MarkerFaceColor='k')
hold on 
plot(YValidationD,LineWidth=0.5,Color='r',LineStyle='--',Marker='o',MarkerFaceColor='r') %[[246 173 174]./255,0.5]
ylim([0.8,1.6])
ylabel("FS")
grid on
legend('Predicted FS','Analyzed FS','Error')
set(gca,"FontSize",25,'FontName',"Times New Roman");
%% 


function [grid_slope,vetor_slope,slopeinfo] = genSlopeMatrix(filename1,filename2)
x_min = -20; % Left boundary
x_max = 100; % Right boundary
y_min = x_min; % Bottom boundary
y_max = x_max; % Top boundary
grid_res = 1;


slope = double(importdata(filename1));
dom = slope(1,:); 
layerNum = slope(2,1); 
layerPoint = slope(3:3+layerNum-1,:);
layerProp = slope(3+layerNum-1:end,1:3);
x = nan(layerNum,size(dom,2)/2);
y = nan(layerNum,size(dom,2)/2);
if layerNum ~= 1
    for j = 1:layerNum
        count = 1;
        if j ~= 1
            jj = j+1;
        else
            jj = j;
        end
        slopesubdata = slope(jj,:);
        for i = 1:2:size(slopesubdata,2)
            x(j,count) = slopesubdata(i);
            y(j,count) = slopesubdata(i+1);
            count = count + 1;
        end
    end
else
    count = 1;
    for i = 1:2:size(dom,2)
        x(1,count) = dom(1,i);
        y(1,count) = dom(1,i+1);
        count = count + 1;
    end
end


PORlist = [];
if nargin>1
    PORlist = double(importdata(filename2));
end


slopeinfo = {max(y(1,:)),atand(y(1,4)/x(1,4)),layerProp};


vetor_slope = [];

grid_coord = [];
for j = floor(y_max+grid_res/2): -grid_res: floor(y_min+grid_res/2+1)
    line = zeros(1,size(x_min:x_max-1,2),2);
    count = 1;
    for i = floor(x_min+grid_res/2): grid_res: floor(x_max+grid_res/2-1)
        line(1,count,1) = i;
        line(1,count,2) = j;
        count = count + 1;
    end
    grid_coord = [grid_coord;line];
end



grid_rho = zeros(size(x_min:x_max-1,2),size(x_min:x_max-1,2));
grid_coh = zeros(size(x_min:x_max-1,2),size(x_min:x_max-1,2));
grid_phi = zeros(size(x_min:x_max-1,2),size(x_min:x_max-1,2));
grid_por = zeros(size(x_min:x_max-1,2),size(x_min:x_max-1,2));
for k = 1:layerNum
    if layerNum ~= 1
        if k == 1
            xv = [x(k+1,:)];
            yv = [y(k+1,:)];
            for poi = 1:size(x,2)
                xt = x(:,end:-1:1);
                yt = y(:,end:-1:1);
                if min(xt(k+1,:))<=xt(k,poi) && xt(k,poi)<=max(xt(k+1,:))
                    if min(yt(k+1,:))<=yt(k,poi) %&& yt(k,poi)<=max(yt(k+1,:))
                        xv = [xv,xt(k,poi)];
                        yv = [yv,yt(k,poi)];
                    end
                end
            end
        elseif k == layerNum
            xv = [x(end,:)];
            yv = [y(end,:)];

            xvp = [xv,xv(xv==max(xv)) + x_max];
            yvp = [yv,yv(xv==max(xv))];
            xvp = [xvp,xvp(end)];
            yvp = [yvp,yvp(end)-y_max];
            xvp = [xvp,xv(xv==min(xv)) - x_max];
            yvp = [yvp,yvp(end)];
            xvp = [xvp,xv(xv==min(xv)) - x_max];
            yvp = [yvp,yv(xv==min(xv))];

            xvp(isnan(xvp))=[]; yvp(isnan(yvp))=[];

            xvv = [];
            yvv = [];
            xt = x(:,end:-1:1);
            yt = y(:,end:-1:1);
            for poi = 1:size(x,2)
                [in,on] = inpolygon(xt(1,poi),yt(1,poi),xvp,yvp);
                if in==1 || on==1
                    xvv = [xvv,xt(1,poi)];
                    yvv = [yvv,yt(1,poi)];
                end
            end


            xv = [xv,xvv];
            yv = [yv,yvv];
            xv(isnan(xv))=[]; yv(isnan(yv))=[];


            Point = [xv;yv];
            d = [xv-mean(xv);yv-(x_min+5)]; 
            angle = atan2(d(2,:),d(1,:)); 
 
            [~, anglei] = sort(angle);   
            Point = Point(:,anglei); 
            Point = [Point Point(:,1)]; 
            xv = Point(1,:);
            yv = Point(2,:);

        else
            xvsub = x(k+1,:);
            yvsub = y(k+1,:);
            [~,idx] = sort(xvsub,'descend');
            xvsub = xvsub(idx);
            yvsub = yvsub(idx);
            xv = [x(k,:)];
            yv = [y(k,:)];

            
            xvv = [];
            yvv = [];
            for poi = 1:size(x,2)
                if min([min(x(k,:)), min(x(k+1,:))])<=x(1,poi) && x(1,poi)<=max([min(x(k,:)), min(x(k+1,:))]) && min([min(y(k,:)), min(y(k+1,:))])<=y(1,poi) && y(1,poi)<=max([min(y(k,:)), min(y(k+1,:))])
                    xvv = [xvv, x(1,poi)];
                    yvv = [yvv, y(1,poi)];
                end                
            end
            xv = [xvv, xv];
            yv = [yvv, yv];
                
            
            xvv = [];
            yvv = [];
            for poi = 1:size(x,2)
                if min([max(x(k,:)), max(x(k+1,:))])<=x(1,poi) && x(1,poi)<=max([max(x(k,:)), max(x(k+1,:))]) && min([max(y(k,:)), max(y(k+1,:))])<=y(1,poi) && y(1,poi)<=max([max(y(k,:)), max(y(k+1,:))])
                    xvv = [xvv, x(1,poi)];
                    yvv = [yvv, y(1,poi)];
                end                
            end
            xv = [xv, xvv, xvsub];
            yv = [yv, yvv, yvsub];
        end
    else
        xv = x;
        yv = y;
    end

    xv(isnan(xv))=[]; yv(isnan(yv))=[];


    for i = 1:size(x_min:x_max-1,2)
        for j = 1:size(x_min:x_max-1,2)
            [in,on] = inpolygon(grid_coord(i,j,1),grid_coord(i,j,2),xv,yv);
            if in == 1 % && on == 0
                grid_rho(i,j) = layerProp(k,1);
                if layerProp(k,2) == 0
                    grid_coh(i,j) = 0.1;
                else
                    grid_coh(i,j) = layerProp(k,2);
                end
                grid_phi(i,j) = layerProp(k,3);
                
                if size(PORlist,2)>1
                    dist = ((PORlist(:,1)-grid_coord(i,j,1)).^2+(PORlist(:,2)-grid_coord(i,j,2)).^2).^0.5;
                    [~,idx] = sort(dist,'ascend');
                    grid_por(i,j) = mean(PORlist(idx(1:4),3));
                end
            end
        end
    end
end

grid_slope(:,:,1) = grid_rho;
grid_slope(:,:,2) = grid_coh;
grid_slope(:,:,3) = grid_phi;
grid_slope(:,:,4) = grid_por;
end 

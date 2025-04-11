% pre_process the ym_actuator_net data from .txt to .csv
data_name = "robot_data_04_11_13_24";

separate_Ankle_from_HipKnee = false;

data = load(data_name+".txt");

%%
if 0

figure
plot([tau_est(:,1),tau_cal(:,1)])

figure
plot([tau_est(:,2),tau_cal(:,2)])

figure
plot([tau_est(:,3),tau_cal(:,3)])

figure
plot([tau_est(:,4),tau_cal(:,4)])

figure
plot([tau_est(:,5),tau_cal(:,5)])

figure
plot([tau_est(:,6),tau_cal(:,6)])
end



%% 查看合法数据
valid_start = 1;
valid_end = 15000;  % 大于15000有偏差

data = data(valid_start:valid_end,:);

tau_est = data(:,37:48);
joint_pos = data(:,13:24);
joint_pos_target = data(:,1:12);
joint_vel = data(:,25:36);
Kp = data(1,49:60);
Kd = data(1,61:72);

tau_cal = Kp.*(joint_pos_target-joint_pos)+Kd.*-joint_vel;

%% 写入数组到csv表格,覆盖写入
prefixes = {'tau_cal', 'tau_est', 'joint_pos', 'joint_pos_target', 'joint_vel'};

%%

if ~separate_Ankle_from_HipKnee
    nj = 12; % number of joints
    Name = {}; 
    for i = 1:numel(prefixes)
        for j = 0:nj-1
            new_entry = sprintf('%s_%d', prefixes{i}, j);  % 格式化字符串
            Name = [Name, new_entry];                      
        end
    end

    data_csv_all = [tau_cal,tau_est,joint_pos,joint_pos_target,joint_vel];
    outputdata_all = array2table(data_csv_all,'VariableNames',Name);
    writetable(outputdata_all,[data_name+".csv"]);

else
    nj = 8; % number of joints
    Name = {}; 
    for i = 1:numel(prefixes)
        for j = 0:nj-1
            new_entry = sprintf('%s_%d', prefixes{i}, j);  % 格式化字符串
            Name = [Name, new_entry];                      
        end
    end
    data_csv_HipKnee = [tau_cal(:,[1:4,7:10]),tau_est(:,[1:4,7:10]),joint_pos(:,[1:4,7:10]),joint_pos_target(:,[1:4,7:10]),joint_vel(:,[1:4,7:10])];
    
    outputdata_HipKnee = array2table(data_csv_HipKnee,'VariableNames',Name);
    writetable(outputdata_HipKnee,[data_name+"_HipKnee.csv"]);
    
    nj = 4; % number of joints
    Name = {}; 
    for i = 1:numel(prefixes)
        for j = 0:nj-1
            new_entry = sprintf('%s_%d', prefixes{i}, j);  % 格式化字符串
            Name = [Name, new_entry];                      
        end
    end
    data_csv_Ankle = [tau_cal(:,[5:6,11:12]),tau_est(:,[5:6,11:12]),joint_pos(:,[5:6,11:12]),joint_pos_target(:,[5:6,11:12]),joint_vel(:,[5:6,11:12])];
    outputdata_Ankle = array2table(data_csv_Ankle,'VariableNames',Name);
    writetable(outputdata_Ankle,[data_name+"_Ankle.csv"]);
end



%%

% Name={'tau_cal_1','tau_est_1','joint_pos_1','joint_pos_target_1','joint_vel_1'};%标题行
% data_csv=[out.tau_cal_1.Data(:), out.tau_est_1.Data(:), out.joint_pos_1.Data(:), out.joint_pos_target_1.Data(:), out.joint_vel_1.Data(:)];%数据
% outputdata = table(data_csv(:,1),data_csv(:,2),data_csv(:,3),data_csv(:,4),data_csv(:,5),'VariableNames',Name);
% writetable(outputdata,"test.csv")

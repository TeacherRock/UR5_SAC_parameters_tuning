clc; clear; close all;
%%
Data = load("command_joint.txt");

for i = 1 : 8
    PCmd(:, :, i) = Data(:, 6*(i-1)+1:6*(i-1)+1+5);
end

t = 0.001 : 0.001 :0.001*length(Data(:, 1));

for i = 1 : 8
    figure('Name', "Solution "+string(i))
    for j = 1 : 6
        subplot(3, 2, j)
        plot(t, PCmd(:, j, i))
        title("Axis" + string(j))
        xlabel("time (s)"); ylabel("position (rad)")
    end
end
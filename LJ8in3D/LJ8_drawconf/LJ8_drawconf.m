function LJ8_drawconf()
close all
fname = "LJ8min_xyz.txt";
data = load(fname);
for j = 1 : 8
    X = data(j*3-2:j*3,:);
    figname = sprintf("LJ8min%i.fig",j);
    DrawConf_BallsStix_Bicolor(X,figname);
end
end
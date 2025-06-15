function drawconf_CV()
close all
list = readmatrix("Figures/MLCVsortCNum_list.csv");
Nconfs = size(list,1);
for j = 1 : Nconfs
    CV1 = list(j,1);
    CV2 = list(j,2);
    drawconf_CVs(CV1,CV2)
end
end
%%
function drawconf_CVs(CV1,CV2)
N1 = 129;
N2 = 129;
h1 = 2.3315875305e-02;
h2 = 3.6207545266e-02;
CV1min = -1.2049260393e+00;
CV1max = 1.7795059996e+00;
CV2min = -2.6719451760e+00;
CV2max = 1.9626206180e+00;

CV1grid = linspace(CV1min,CV1max,N2);
CV2grid = linspace(CV2min,CV2max,N1);

j1 = floor((CV1 - CV1min)/h1)+1;
j2 = floor((CV2 - CV2min)/h2)+1;

% 	for( i=0; i<N1; i++ ) {
% 		for( j=0; j<N2; j++ ) {
% 			ind = (i + j*N1)*dim;
% 			for( n = 0; n < dim; n++ ) {
% 				fprintf(fconf,"%.4e\t",bin_confs[ind + n]);
% 			}
% 			fprintf(fconf,"\n");
% 		}
% 	}

conf = load("Data/LJ8bins_confs.txt");
ind = j2 + j1*N2;
xyz = reshape(conf(ind,:)',[8,3])';
figname = sprintf("Figures/Confs/Conf_CV1_%.3f_CV2_%.3f.fig",CV1,CV2);
DrawConf_BallsStix_Bicolor(xyz,figname);
% bins = load("Data/LJ7mu2mu3_wmetad_bins.txt");
% 
% fprintf("mu2 = %d, mu3 = %d, bins = %d\n",mu2,mu3,bins(j2,j1));
% for j = 1 : 7
%     fprintf("%d\t%d\n",xy(j),xy(j+7));
% end

% figure; hold on;
% t = linspace(0,2*pi,100);
% r = 0.5*2^(1/6);
% for j = 1 : 7
%     plot(xy(j)+r*cos(t),xy(j+7)+r*sin(t),'Linewidth',4,'color',[0,0,0.5]);
% end
% daspect([1,1,1])
% axis off

% figname = sprintf("Figures/Confs/Conf_CV1_%.3f_CV2_%.3f.eps",CV1,CV2);
% saveas(gcf,figname,'epsc')



end

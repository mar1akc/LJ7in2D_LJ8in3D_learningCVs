function LJ7mesh_MargotCV2D()
close all
BETA = 9;
dirname = "/Users/mariacameron/Dropbox/Work/My_Programs/LJ7in2D/";
subdirname = sprintf("CV_OrderedCoordNum2/FEMdataBETA%i/",BETA);
dirname = strcat(dirname,subdirname);
pA = readmatrix(strcat(dirname,'A_bdry.csv'));
NpA = length(pA);
NA = NpA/2;
pA = (reshape(pA',[2,NA]))';
pA(end,:) = [];
%
pB = readmatrix(strcat(dirname,'B_bdry.csv'));
NpB = length(pB);
NB = NpB/2;
pB = (reshape(pB',[2,NB]))';
pB(end,:) = [];

%
p0 = readmatrix(strcat(dirname,'O_bdry.csv'));
Np0 = length(p0);
N0 = Np0/2;
p_outer = (reshape(p0',[2,N0]))';
p_outer(end,:) = [];

figure(1);
hold on; grid;
plot(pA(:,1),pA(:,2),'Linewidth',2);
plot(pB(:,1),pB(:,2),'Linewidth',2);
plot(p_outer(:,1),p_outer(:,2),'Linewidth',2);

Nouter = size(p_outer,1);
Na = size(pA,1);
Nb = size(pB,1);
ea = bdry_connect(1:Na);
eb = bdry_connect(Na+1:Na+Nb);
e_outer = bdry_connect(Na+Nb+1:Na+Nb+Nouter);
bdry = [pA;pB;p_outer];
bdryconnect = [ea;eb;e_outer];

    opts.kind = 'delaunay';
    opts.rho2 = +1.00 ;

   [pts,etri, ...
    tr,tnum] = refine2(bdry,bdryconnect,[]  ,opts) ;
[pts,etri,tr,tnum] = smooth2(pts,etri,tr,tnum);
[pts,etri,tr,tnum] = tridiv2(pts,etri,tr,tnum);
[pts,etri,tr,tnum] = smooth2(pts,etri,tr,tnum);


triplot(tr,pts(:,1),pts(:,2));

fprintf("Npts = %d, Ntri = %d\n",size(pts,1),size(tr,1));
%%
%% extract boundary nodes and boundary edges
% Find boundary nodes
edges = [tr(:,1),tr(:,2);tr(:,2),tr(:,3);tr(:,1),tr(:,3)];
e0 = edges;
edges = sort(edges,2);
[esort,isort2] = sort(edges(:,2),'ascend');
edges = edges(isort2,:);
[esort,isort1] = sort(edges(:,1),'ascend');
edges = edges(isort1,:);
Nedges = length(edges);
fprintf('N edges = %i, N points = %i\n',length(edges),length(pts));
% boundary edges are encountered only once as they belong to only one
% triangle
[uedges,ishrink,iexpand] = unique(edges,'rows');
Nis = length(ishrink);
fprintf('N ishrink = %i\n',Nis);
gap = circshift(ishrink,[-1,0]) - ishrink;
if ishrink(end) == Nedges
    gap(end) = 1;
else
    gap(end) = [];
end    
i1 = find(gap == 1);
n1 = length(i1);
i2 = find(gap == 2);
n2 = length(i2);
fprintf('n1 = %i, n2 = %i\n',n1,n2);
ie1 = ishrink(i1);
bedges = edges(ie1,:);
Nbe = size(bedges,1);
ind = [1 : Nedges]';
ie2 = ind;
ie2(ie1) = [];
Nie = length(ie2);
fprintf('Nbe = %i, Nie = %i\n',Nbe,Nie);
figure; hold on
for i = 1 : Nbe
    j = bedges(i,:);
    plot([pts(j(1),1),pts(j(2),1)],[pts(j(1),2),pts(j(2),2)],'Linewidth',2);
end
bb = [bedges(:,1);bedges(:,2)];
ibnodes = unique(bb);
Nib = length(ibnodes);
bnodes = pts(ibnodes,:);
plot(bnodes(:,1),bnodes(:,2),'.','Markersize',10);
drawnow;
Nbdry = size(bnodes,1);
% To use Tarjan's algorithm, we need the vertices to be indexed from 1 to
% Nverts
Npts = size(pts,1);
map = zeros(Npts,1);
map(ibnodes) = (1 : Nib)';
E = [map(bedges(:,1)),map(bedges(:,2));map(bedges(:,2)),map(bedges(:,1))];
% Tarjan's algorithm
NSCC = Tarjan((1 : Nib)',E,dirname);
col = parula(NSCC + 1);
for i = 1 : NSCC
    fname = strcat(dirname,sprintf('/LJ7_SCC%d',i));
    SCC = load(fname);
    ind = SCC.SCC;
    plot(pts(ibnodes(ind),1),pts(ibnodes(ind),2),'.','Markersize',10,'color',col(i,:));
    drawnow;
    fprintf('Boundary: SCC %i: %i nodes\n',i,length(ind));
end
daspect([1,1,1])
set(gca,'Fontsize',20);
triplot(tr,pts(:,1),pts(:,2));

% fprintf("B: Npts = %d, Ntri = %d\n",size(pts,1),size(tr,1));
%     fname_read = strcat(dirname,sprintf('/LJ7_SCC%d.mat',1));    
%     SCC = load(fname_read);
%     C = ibnodes(SCC.SCC);
% plot(pts(C,1),pts(C,2),'.','color','r','markersize',20);
% for k = 1 : length(C)
%     fprintf("%d\t%d\t%d\t%d\n",k,C(k)-1,pts(C(k),1),pts(C(k),2));
% end
% 
%% write the results into files
writematrix(pts,strcat(dirname,'/pts.csv'))
writematrix(tr-ones(size(tr)),strcat(dirname,'/tri.csv'))
fname_writeA = strcat(dirname,'/Abdry_idx.csv');
fname_writeB = strcat(dirname,'/Bbdry_idx.csv');
for j = 1 : 2
    fname_read = strcat(dirname,sprintf('/LJ7_SCC%d.mat',j));    
    SCC = load(fname_read);
    C = ibnodes(SCC.SCC) - 1;
    if j == 1
        fname_write = fname_writeA;
    else
        fname_write = fname_writeB;
    end
    fprintf("j = %d, size(SCC) = %d\n",j,length(C));
    writematrix(C,fname_write);
end
%% Triangulate sets A and B
figure;
hold on;
[pts,etri,tr,tnum] = refine2(pA,ea,[]  ,opts) ;
[pts,etri,tr,tnum] = smooth2(pts,etri,tr,tnum);
[pts,etri,tr,tnum] = tridiv2(pts,etri,tr,tnum);
[pts,etri,tr,tnum] = smooth2(pts,etri,tr,tnum);
% [pts,tr] = refine(pts,tr);
% [pts,tr] = smoothmesh(pts,tr);
triplot(tr,pts(:,1),pts(:,2));
writematrix(pts,strcat(dirname,'/ptsA.csv'));
writematrix(tr-ones(size(tr)),strcat(dirname,'/triA.csv'))
fprintf("A: Npts = %d, Ntri = %d\n",size(pts,1),size(tr,1));

%
figure;
hold on;
eb = bdry_connect(1:Nb);
[pts,etri,tr,tnum] = refine2(pB,eb,[]  ,opts) ;
[pts,etri,tr,tnum] = smooth2(pts,etri,tr,tnum);
[pts,etri,tr,tnum] = tridiv2(pts,etri,tr,tnum);
[pts,etri,tr,tnum] = smooth2(pts,etri,tr,tnum);
% [pts,tr] = refine(pts,tr);
% [pts,tr] = smoothmesh(pts,tr);
triplot(tr,pts(:,1),pts(:,2));
writematrix(pts,strcat(dirname,'/ptsB.csv'));
writematrix(tr-ones(size(tr)),strcat(dirname,'/triB.csv'));
end
%%
function bedges = bdry_connect(inds)
N1 = length(inds);
inds = reshape(inds,[N1,1]);
bedges = [inds,circshift(inds,[-1,0])];
end

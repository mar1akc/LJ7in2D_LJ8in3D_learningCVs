%%
function DrawConf_BallsStix_Bicolor(x,figname)
figure;
hold on
LJrad = 2^(1/6);
tol = 5e-2;
N = 8;
A = zeros(N);

for j = 1 : 8
    d = x(:,j)*ones(1,N) - x;
    d = sqrt(sum(d.^2,1));
    A(j,abs(d-LJrad)<tol) = 1;
end
N = 8;

flag = 1;

if flag == 1
    rstar = 0.2; % radius of ball
    cr = 0.06; % radius of cylinder
    alim = 1.5;
    %% Stuff for graphics
    xyzlim = alim - 1.5*rstar;
    col = jet(N);

    t = linspace(-0.5,0.5,50);
    [X,Y,Z] = meshgrid(t,t,t);
    V = sqrt(X.^2 + Y.^2 + Z.^2);
    [bfaces,bverts] = isosurface(X,Y,Z,V,rstar); % faces and vertices of balls
    nbverts = size(bverts,1);
    eb = ones(nbverts,1);
    V = sqrt(X.^2 + Y.^2);
    [cfaces,cverts] = isosurface(X,Y,Z,V,cr); % faces and vertices of balls
    ncverts = size(cverts,1);
    ec = ones(ncverts,1); 

    ifig = 1;

    [m,N] = size(x);


    ind = find(A == 1);
    [I J] = ind2sub([N,N],ind);
    ii = find(I > J);
    [I J] = ind2sub([N,N],ind(ii));
    Nbonds = length(ii);

    cbond = 0*[ones(Nbonds,1),ones(Nbonds,1),ones(Nbonds,1)];

    cx = sum(x,2)/N;
    x = x - cx*ones(1,N);
    for j = 1 : N 
        xyz = x(:,j);
        patch('Vertices',bverts + eb*xyz','Faces',bfaces,'Facecolor','b','EdgeColor','none','LineStyle','none');
    end
    for i = 1 : Nbonds
        x0 = x(:,I(i));
        x1 = x(:,J(i));
        u = x1 - x0;
        u = u/norm(u); % unit vector parallel to the the vector x0 --> x1
        R = make_rotation_matrix(u); % makes rotation matrix 
        % for turn around axis normal to Oz cross u by angle between (Oz,u)
        cv = cverts*R';
        cv = cv + 0.5*ec*(x0+x1)'; % shift start of the cylinder to 0.5*(x0 + x1);
        patch('Vertices',cv,'Faces',cfaces,'Facecolor',cbond(i,:),'EdgeColor','none','LineStyle','none');
    end

    view(3)
    set(gca,'DataAspectRatio',[1,1,1]);
    camlight('headlights')
    lighting gouraud
    alpha(0.8)
    set(gca,'DataAspectRatio',[1 1 1]);
    axis off
    drawnow;
end
saveas(gcf,figname,'epsc')

end

%%
%%
function R = make_rotation_matrix(u)
I = eye(3);
e = [0;0;1];
co = u'*e;
u = cross(e,u);
si = norm(u);
if si < 1e-12
    if co > 0
        R = I;
    else
        R = I;
        R(3,3) = -1;
    end
    return
end
u = u/si;    
ucr = [0,-u(3),u(2);u(3),0,-u(1);-u(2),u(1),0];
uu = u*u';
R = I*co + ucr*si + (1 - co)*uu;
end



# function [ Lx ] = lagrange(xi , yi , x)
# % LAGRANGE	Polyn�me de Lagrange passant par les points xi et yi
# %
# % Syntaxe: [ Ly ] = lagrange(xi , yi , x)
# %
# % Argument d'entree
# %	xi		-	Vecteur contenant les abscisses des points d'interpolation
# %	yi		-	Vecteur contenant les ordonnees des points d'interpolation
# %	x		-	Vecteur contenant les points o� le polyn�me de Lagrange 
# %				sera evalue
# %
# % Arguments de sortie
# %	Lx		-	Vecteur contenant les valeurs du polyn�mes aux points x
# %
# % Exemples d'appel
# %	[ Lx ] = lagrange([-1,0,1] , [1,0,1] , linspace(-1,1,200))
# %
# % Le calcul du polynome interpolant est base sur la formule barycentrique ;
# % une variation stable de la formule d'interpolation de Lagrange vue au
# % cours. Cette implementation est basee sur celle de Greg von Winckel,
# % elle-meme basee sur l'article de Berrut et Trefethen (SIAM Review, 2004).
# %
# % Normalement, on devrait utiliser une version particularisee de cette
# % fonction si le vecteur xi donne les noeuds d'interpolation de Chebychev.
# %
# % "Barycentric interpolation is a variant of Lagrange polynomial
# % interpolation that is fast and stable. It deserves to be known as the
# % standard method of polynomial interpolation." (Berrut and Trefethen, 2004).
# %
# % Cree par D. Orban, 2009. 
# % Modifie par A. Paquette-Rufiange, 2018.
# %

# %% Verification des arguments d'entrees
# if ~isnumeric(xi) || ~isvector(xi)
# 	error('Les abscisses xi ne sont pas arrangees en vecteur')
# elseif ~isnumeric(yi) || ~isvector(yi)
# 	error('Les ordonnees yi ne sont pas arrangees en vecteur')
# elseif length(xi) ~= length(yi)
# 	error('Les vecteurs xi et yi doivent avoir la meme taille');
# elseif ~isnumeric(x) || ~isvector(x)
# 	error('Les points o� l''on evalue le polynome doivent etre arrangees en vecteur');
# elseif isequal(xi,x)
# 	warning('Le polyn�me d''interpolation est evalue exactement au points d''interpolation')
# end


# %% Calcul de l'interpolant aux points x

# M = length(xi);  xxi = reshape(xi,M,1);  % Donnees en vecteur colonne
# N = length(x);   xx = reshape(x,N,1);    % Grille fine en vecteur colonne

# Xi = repmat(xxi,1,M);
# W = repmat(1./prod(Xi-Xi.'+eye(M),1),N,1);   % Poids barycentriques
# xdist=repmat(xx,1,M)-repmat(xxi.',N,1);   % Distances aux points d'interpolation

# % On place des NaN aux noeuds d'interpolation pour eviter la division par zero
# [fixi,fixj]=find(xdist==0);
# xdist(fixi,fixj)=NaN;
# H=W./xdist;

# Lx=(H * reshape(yi,M,1))./sum(H,2);  % Valeurs du polynome sur la grille fine
# Lx(fixi)=yi(fixj);        % On restaure les valeurs exacte a la place des NaN
# if (size(Lx) ~= size(x))
#   Lx = Lx';
# end


# function [ Sx ] = splinec( xi , yi , x , type_S , val_S)
# % SPLINEC	Spline cubique passant par les points xi et yi avec differents 
# %			type de conditions frontieres
# %
# % Syntaxe: [ Sx ] = splinec( xi , yi , x , type_f , val_f)
# %
# % Argument d'entree
# %	xi		-	Vecteur contenant les abscisses des points d'interpolation
# %	yi		-	Vecteur contenant les ordonnees des points d'interpolation
# %	x		-	Vecteur contenant les points o� la spline sera evalue
# %	type_S	-	Vecteur de dimension 2 contenant le type des conditions 
# %				frontieres imposees en x0 et xn. Les choix possibles sont:
# %					[1,1] -> Spline naturelle 
# %					[2,2] -> Spline avec courbure prescrite
# %					[3,3] -> Spline avec courbure constante
# %					[4,4] -> Spline avec pente prescrite
# %					[i,j] -> Spline avec condition i imposee en x0 et 
# %							 condition j imposee en xn
# %							 	
# %	val_S	-	Vecteur de dimension 2 contenant les deux conditions 
# %				limites imposees en x0 et xn. Les choix possibles sont:
# %					- Si type_S(1) = 1 ou 3, alors val_S(1) = nan
# %					- Si type_S(1) = 2 ou 4, alors val_S(1) = a, o� a 
# %					  represente resp. la courbure ou la pente en x0
# %					- Si type_S(2) = 1 ou 3, alors val_S(1) = nan
# %					- Si type_S(2) = 2 ou 4, alors val_S(1) = b, o� b 
# %					  represente resp. la courbure ou la pente en xn
# %
# % Arguments de sortie
# %	Sx		-	Vecteur contenant les valeurs de la spline aux points x
# %
# % Exemples d'appel
# %	[ Sx ] = splinec([1,2,4,5], [1,9,2,11], linspace(1,5), [1,1] , [nan,nan])
# %	[ Sx ] = splinec([1,2,4,5], [1,9,2,11], linspace(1,5), [2,2] , [5,-6])
# %	[ Sx ] = splinec([1,2,4,5], [1,9,2,11], linspace(1,5), [3,3] , [nan,nan])
# %	[ Sx ] = splinec([1,2,4,5], [1,9,2,11], linspace(1,5), [4,4] , [-30,-10])
# %	[ Sx ] = splinec([1,2,4,5], [1,9,2,11], linspace(1,5), [3,4] , [nan,-10])


# %% Verification des arguments d'entrees
# if ~isnumeric(xi) || ~isvector(xi)
# 	error('Les abscisses xi ne sont pas arrangees en vecteur')
# elseif ~isequal(sort(xi),xi)
# 	error('Les abscisses ne sont pas arrangees en ordre croissant')
# elseif ~isnumeric(yi) || ~isvector(yi)
# 	error('Les ordonnees yi ne sont pas arrangees en vecteur')
# elseif length(xi) ~= length(yi)
# 	error('Les vecteurs xi et yi doivent avoir la meme taille');
# elseif ~isnumeric(x) || ~isvector(x)
# 	error('Les points o� l''on evalue la spline cubique doivent etre arrangees en vecteur');
# elseif isequal(xi,x)
# 	warning('Le polyn�me d''interpolation est evalue exactement au points d''interpolation')
# elseif ~isnumeric(type_S) || ~isvector(type_S) || length(type_S)~=2 || ~any(type_S(1)==[1,2,3,4]) || ~any(type_S(2)==[1,2,3,4])
# 	error('Les types de conditions frontieres ne sont pas valide')
# elseif ~isnumeric(val_S) || ~isvector(val_S) || length(val_S)~=2
# 	error('Les valeurs des conditions frontieres ne sont pas dans un vecteur de dimension 2')
# end


# %% Calcul des coefficients S''

# % Assemblage de la matrice
# nb_f	=	length(xi);
# h		=	reshape(diff(xi),1,[]);
# denom	=	h(1:end-1)+h(2:end);

# M			=	[[h(1:end-1)./denom,0,0]',[0;2*ones(nb_f-2,1);0],[0,0,h(2:end)./denom]'];
# matrice		=	spdiags(M,[-1,0,1],nb_f,nb_f);

# % Calcul des 2eme differences divisees
# mat_diff_div1	=	diff_div(xi,yi,1);
# mat_diff_div2	=	diff_div(xi,mat_diff_div1,2);

# % Terme de droite
# B	=	zeros(nb_f,1);
# B(2:end-1)	=	6*mat_diff_div2;

# % Imposition des conditions frontieres
# switch type_S(1)
# 	case 1
# 		matrice(1,1)		=	1;
# 		B(1)				=	0;
# 	case 2
# 		matrice(1,1)		=	1;
# 		B(1)				=	val_S(1);
# 	case 3
# 		matrice(1,[1,2])		=	[1,-1];
# 		B(1)					=	0;
# 	case 4
# 		matrice(1,[1,2])		=	[2,1];
# 		B(1)					=	6/h(1)   * ((yi(2) - yi(1))/h(1) - val_S(1));
# end

# switch type_S(2)
# 	case 1
# 		matrice(nb_f,nb_f)	=	1;
# 		B(end)				=	0;
# 	case 2
# 		matrice(nb_f,nb_f)	=	1;
# 		B(end)				=	val_S(2);
# 	case 3
# 		matrice(end,[end-1,end])=	[-1,1];
# 		B(end)					=	0;
# 	case 4
# 		matrice(end,[end-1,end])=	[1,2];
# 		B(end)					=	6/h(end) * (val_S(2) - (yi(end) - yi(end-1))/h(end));
# end

# % Resolution du systeme lineaire
# Spp		=	matrice\B;


# %% Calcul de la spline aux points x 

# Sx	=	nan(size(x));

# for t=1:nb_f-1
# 	x_inter		=	(x>= xi(t)) & (x <= xi(t+1));
# 	Sx(x_inter) =	poly_spline(xi(t:t+1),yi(t:t+1),Spp(t:t+1),h(t),x(x_inter));
# end

# end


# function [ f ] = diff_div(x,y,n)
# 	diff_x	=	reshape(x(n+1:end) - x(1:end-n),1,[]);
# 	diff_y	=	reshape(diff(y),1,[]);
# 	f		=	diff_y./diff_x;
# end


# function [ Px ] = poly_spline(xi,yi,Spp,h,x)
# 	Px	=	-Spp(1)/(6*h)*(x-xi(2)).^3 + Spp(2)/(6*h)*(x-xi(1)).^3 - ...
# 			 (yi(1)/h - Spp(1)*h/6)*(x-xi(2)) + (yi(2)/h - Spp(2)*h/6)*(x-xi(1));
# end
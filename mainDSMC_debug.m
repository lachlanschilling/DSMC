 % DSMC SOLVER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FILE:         mainDSMC_debugMode.m
% AUTHOR:       Lachlan Schilling
% DATE (D/M/Y): 06/10/2022
% DESCRIPTION:  Main function for the DSMC solver. Inputs can be adjusted
%               below (number of simulations) and in the initSIMU.m file. 
%_________________________________________________________________________%



% VARIABLES & HOUSEKEEPING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clearvars
close all
clear global
global k T0 TW m d L dL debugMode leftW rightW bottomW topW frontW backW...
       colCELL n volDOMA volCELL N_dt N_dt_tau coeffNTC sigma dt tau beta...
       nPART nPARTeff lambda colsP colsW particle vRELMAX Nremcol nCELL...
       samples printInfo save steps attempt s doTiles;
debugMode = false;
printInfo = true;
attempt   = 1;
movie     = true;
doTiles   = false;
sims      = 1;
%_________________________________________________________________________%



% SIMULATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tic
for s = 1:sims                          % for all simulations...
	initSIMU();                         % set up current simulation
	indxPART();                         % place particles in cells
    for steps = 1 : N_dt                % for all time steps...
        advcPART();                     % advect & collide particles
        sampCELL();                     % sample the data
        saveDATA();                     % save the data
    end
end
%_________________________________________________________________________%
toc
if movie 
    doMovie();                          % create a movie from saved data
end   
tiledPlot();


% FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function initSIMU()
% DSMC SOLVER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FILE:         initSIMU.m
% AUTHOR:       Lachlan Schilling
% DATE (D/M/Y): 06/10/2022
% DESCRIPTION:  Variable initialisation function for the DSMC solver. 
%               The flow direction is z.
%_________________________________________________________________________%



% DEFINE GLOBALS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global k T0 TW m d L dL debugMode leftW rightW bottomW topW frontW backW...
       colCELL n volDOMA volCELL N_dt N_dt_tau coeffNTC sigma dt tau beta...
       nPART nPARTeff lambda colsP colsW particle vRELMAX Nremcol nCELL...
       samples save s;
%_________________________________________________________________________%



% INPUTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
T0      = 293;          % K     Initial particle temperature
TW      = 1000;         % K     Wall temperature
m       = 6.63e-26;     % kg    Argon particle mass
d       = 3.66e-10;     % m     Argon particle diameter
k       = 1.380649e-23; % J/K   Boltzmann's constant 
L       = 1e-3*[1 1 1]; % m     Domain size (1 mm3)
colCELL = [10 10 10];   % -     Number of collision cells in x y z
dL      = L./colCELL;   % -     Dimensions of cells
n       = 1e18;         % /m3   Molecules per cubic meter   1e18
TMAX    = 1.3*TW;       % K     Temperature guess for finding max velocity
MBP     = 1e-6;         % -     Probability cut off for vMAX (1 in million)
nPART   = 50000;        % -     Number of particles in the simulation.
N_dt    = 6e3;          % -     Total number of time steps. 6000 is good
N_dt_tau= 5e6;          % -     Number of timesteps per mean collision time
beta    = sqrt(k / m);  % ?     Variable for efficient computation - IGNORE
%_________________________________________________________________________%



% MEAN FREE PATH VERIFICATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lambda  = 1/(sqrt(2) * pi * d^2 * n);   % Mean free path
if sum(lambda > L./colCELL) > 0
    fprintf('Mean free path is larger than the cell sizes.\n')
    if L(1) == 1e-3
        fprintf('\tMean Free Path:\t%0.4f mm\n',lambda*10^3)
        fprintf('\tCell Sizes:\t\t%0.3f mm by %0.3f mm by %0.3f mm\n',...
        dL(1)*10^3, dL(2)*10^3, dL(3)*10^3)
    elseif L(1) == 1e-6
        fprintf('\tMean Free Path:\t%0.4f µm\n',lambda*10^6)
        fprintf('\tCell Sizes:\t\t%0.3f µm by %0.3f µm by %0.3f µm\n',...
        L(1)/colCELL(1)*10^6, L(2)/colCELL(2)*10^6, L(3)/colCELL(3)*10^6)
    end
end
volDOMA     = L(1) * L(2) * L(3);                   % volume of the domain
volCELL     = dL(1)* dL(2)* dL(3);                  % volume of each cell  
nCELL       = colCELL(1) * colCELL(2) * colCELL(3); % total number of cells
nPARTeff    = n*volDOMA/nPART;                      % effective num of atms
fprintf('Each particle represents %i atoms.\n', floor(nPARTeff));
%_________________________________________________________________________%



% MAX VELOCITY DERIVATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vSWEEP  = 0:1:5e3;                                  % range of velocities
mbPopA  = MaxwellBoltzmann(TMAX, vSWEEP, k, m);     % MB dist. of range
if debugMode
    debug1 = figure;
    plot(mbPopA)                                    % MB dist. visual
    xlabel('Index')
    ylabel('MB Proability Distribution')
    close(debug1)
end
mbPopA  = cumsum(mbPopA);                           % cumultive dist.
indx    = find(mbPopA > (1- MBP), 1);               % find index based on
vMAX    = vSWEEP(indx);                             % the prop. cutoff
%_________________________________________________________________________%



% INITIALISE VELOCITIES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
vMP     = sqrt(2 * k * T0 / m);         % Most probable velocity
vAVE    = (2 / sqrt(pi)) * vMP;         % Average velocity
vVEC    = 0:1:vMAX;                     % Velocity array. 1 m/s accuracy
vACT    = zeros(nPART, 1);              % Actual velocities initialisation
if (vMAX < vAVE) || (vMAX > 10000)
    fprintf('Warning. Maximum velocity may be unphysical.')
    fprintf('vMAX = %.1f m/s\n', vMAX)
end
random  = rand(nPART, 1);               % Uniform random numbers for vel
mbPopA  = MaxwellBoltzmann(T0, vVEC, k, m);   
mbPopA  = cumsum(mbPopA);               % Cumulative probabilites
% Actual velocities
for i = 1:nPART
    index = find(random(i) < mbPopA);
    vACT(i) = vVEC(index(1));
end
% Distribute to particles
particle.vel(:,1:2) = (2 * randi(2, [nPART, 2]) - 3) .* vACT/sqrt(3);
particle.vel(:,3)   = vACT/sqrt(3);
% Plot velocity distribution
if debugMode
    debug2 = figure;
    hold on
    plot(vACT)
    plot(ones(1,length(vACT))*vAVE)
    plot(ones(1,length(vACT))*vMAX)
    legend('Velocity','Mean Velocity', 'Max Velocity')
    title("Velocity Distribution (T = " + num2str(T0) + " K)")
    ylabel('Speed (m/s)')
    xlabel('Particle #')
    close(debug2)
end
% Initialise relative velocities w/ educated guess
% i.e. 2x max(abs(vACT))/sqrt(3)
vGuess      = 2 * max(sqrt(vACT.^2),[],'all') / sqrt(3);
vRELMAX     = vGuess * ones(nCELL, 1);
%_________________________________________________________________________%



% INITIALISE POSITIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
particle.pos(:,1) = L(1) * rand(nPART, 1); % X-coordinates of the particles
particle.pos(:,2) = L(2) * rand(nPART, 1); % Y-coordinates of the particles
particle.pos(:,3) = L(3) * rand(nPART, 1); % Z-coordinates of the particles
if debugMode
    q1 = figure;
    plot3(particle.pos(:,1),particle.pos(:,2),particle.pos(:,3),'.');
    xlabel('X (m)')
    ylabel('Y (m)')
    zlabel('Z (m)')
    close(q1);
end
%_________________________________________________________________________%



% INITIALISE REMAINDER COUNTER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Nremcol     = zeros(nCELL, 1);
%_________________________________________________________________________%



% INITIALISE WALLS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
leftW       = struct('r',0,     'dir',1,	'TW',TW);
rightW      = struct('r',L(1),  'dir',-1,	'TW',TW);
bottomW     = struct('r',0,     'dir',1,    'TW',TW);
topW        = struct('r',L(2),  'dir',-1,	'TW',TW);
backW       = struct('r',0,     'dir',1,    'TW',T0);
frontW      = struct('r',L(3),  'dir',-1,	'TW',T0);
%_________________________________________________________________________%



% INITIALISE FLOWTIME %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tau         = lambda / vAVE;        % Mean collision time
dt          = tau / N_dt_tau;       % Time-step
sigma       = pi *d^2;              % Collision cross section
coeffNTC    = 0.5 * sigma * nPARTeff * dt / volCELL;
%_________________________________________________________________________%



% INITIALISE CELL INDEXING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
particle.cell(:,1) = zeros(nPART, 1);
particle.cell(:,2) = 1 : nPART;       % add indexing feature
particle.id        = zeros(nCELL, 2); % add indexing feature
%_________________________________________________________________________%



% INITIALISE SAMPLING DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
samples.vel = zeros(nCELL, 3);
samples.V2  = zeros(nCELL, 1);
samples.T   = zeros(nCELL, 1);
samples.rho = zeros(nCELL, 1);
%_________________________________________________________________________%



% INITALISE SAVE DATA %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if s == 1
    save.vel    = zeros(N_dt, colCELL(1), colCELL(2), 3);
    save.T      = zeros(N_dt, colCELL(1), colCELL(2));
    save.rho    = zeros(N_dt, colCELL(1), colCELL(2));
end
%_________________________________________________________________________%



% INITIALISE COLLISION COUNTERS (debug only) %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if debugMode
    % Walls
    colsW.left      = 0; colsW.right	= 0;
    colsW.bottom	= 0; colsW.top      = 0;
    colsW.back      = 0; colsW.front	= 0;
    % Particles
    colsP = 0;
end
%_________________________________________________________________________%



% MAXWELL-BOLTZMANN DISTRIBUTION
function P = MaxwellBoltzmann(T, v, k, m)
    P = 4 * pi * (m./(2 * pi * k * T)).^1.5 .* v.^2 .* exp(-v.^2 .* ...
        m./(2*k*T));
end   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
function indxPART()
% DSMC SOLVER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FILE:         indxPART.m
% AUTHOR:       Lachlan Schilling
% DATE (D/M/Y): 06/10/2022
% DESCRIPTION:  Particle indexer function for the DSMC solver that places 
%               the particles into cells.
%_________________________________________________________________________%

% GLOBAL VARIABLES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global dL colCELL nPART particle debugMode nCELL L;

% FINDING GRID LOCATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xyzCELL = ceil(particle.pos ./ (ones(nPART, 3) .* dL)); 

% PLACE INTO STRUCTURE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
particle.cell(:,2) = 1 : nPART;     
particle.cell(:,1) = xyzCELL(:,1) + colCELL(1) * (xyzCELL(:,2) - 1) ...
                                  + colCELL(1) * colCELL(2) * ...
                                  (xyzCELL(:,3) - 1);

% SORT PARTICLES BASED ON CELLS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
particle.cell = sortrows(particle.cell,1);  % 1x sorting step / timestep
cl = 1;
particle.id(cl, 1) = 1;                     % first particle
subtot = 1;
for i = 2:nPART                             % 1x index step / timestep
    if particle.cell(i,1) == cl
        subtot = subtot + 1;                % increase num of particles
    else
        particle.id(cl, 2) = subtot;        % particle sum for this cell
        cl = cl + 1;                        % incrememnt cell
        particle.id(cl, 1) = i;             % set index for this cell group
        subtot = 1;                         % reset subtotal
    end
end
particle.id(cl, 2) = subtot;

% VERIFY IN CORRECT CELLS
if debugMode
    q2 = figure;
    hold on
    axis equal
    xlim([0 L(1)])
    ylim([0 L(1)])
    zlim([0 L(1)])
    xlabel('X (m)')
    ylabel('Y (m)')
    zlabel('Z (m)')
    totalParticlesPlotted = 0;
    for c = 1:nCELL
        if mod(c,4) == 0
            colour = 'r';
        elseif mod(c,4) == 1
            colour = 'b';
        elseif mod(c,4) == 2
            colour = 'g';
        else
            colour = 'y';
        end
        indxCELL = particle.id(c, 1) : (particle.id(c, 1) + ...
            particle.id(c, 2) - 1);
        indxPARt = particle.cell(indxCELL, 2);
        plot3(particle.pos(indxPARt, 1), particle.pos(indxPARt, 2),...
            particle.pos(indxPARt, 3), '.','color',colour);
        totalParticlesPlotted = totalParticlesPlotted + particle.id(c, 2);
    end
    axis equal
    hold off
    close all;
end

%_________________________________________________________________________%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
function advcPART()
% DSMC SOLVER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FILE:         advcPART.m
% AUTHOR:       Lachlan Schilling
% DATE (D/M/Y): 06/10/2022
% DESCRIPTION:  Particle advection function for the DSMC solver. 
%               Includes both particle and wall collision function calls.
%_________________________________________________________________________%


% GLOBAL VARIABLES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global dL colCELL nPART particle nCELL L leftW rightW bottomW topW frontW...
       backW dt beta colsW particle debugMode coeffNTC colsP particle ...
       vRELMAX Nremcol;
%_________________________________________________________________________%


% ADVECTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
particleOld  = particle.pos;
particle.pos = particle.pos + particle.vel * dt;
%_________________________________________________________________________%


% COLLISIONS - CELL BY CELL BASIS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bndyCELL(particleOld); % collide particles with boundary
indxPART(); % Reindex particles after hitting the boundaries.
for c = 1:nCELL
    % collide particles with particles
    collCELL(c);
end
%_________________________________________________________________________%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
function bndyCELL(particleOld)
% DSMC SOLVER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FILE:         bndyCELL.m
% AUTHOR:       Lachlan Schilling
% DATE (D/M/Y): 06/10/2022
% DESCRIPTION:  Boundary condition function for the DSMC solver. 
%               Includes both periodic and thermal wall collisions.
%_________________________________________________________________________%



% GLOBAL VARIABLES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global L debugMode leftW rightW bottomW topW frontW backW...
       dt beta colsW particle;
%_________________________________________________________________________%



% FINDING THE PARTICLES COLLIDING WITH THE WALLS %%%%%%%%%%%%%%%%%%%%%%%%%%
lef = particle.pos(:, 1) <= leftW.r;   
rig = particle.pos(:, 1) >= rightW.r;
bot = particle.pos(:, 2) <= bottomW.r; 
top = particle.pos(:, 2) >= topW.r;
bac = particle.pos(:, 3) <= backW.r;   
fro = particle.pos(:, 3) >= frontW.r;
% Total number of particles that have collided with walls
sumlef = sum(lef);      sumrig = sum(rig);
sumbot = sum(bot);      sumtop = sum(top);
sumbac = sum(bac);      sumfro = sum(fro);
%_________________________________________________________________________%



% UPDATING THE WALL COLLISION COUNTERS (debug only) %%%%%%%%%%%%%%%%%%%%%%%
if debugMode
    % Collision counters of each wall
    colsW.left      = colsW.left    + sumlef;      
    colsW.right     = colsW.right   + sumrig;
    colsW.bottom    = colsW.bottom  + sumbot;    
    colsW.top       = colsW.top     + sumtop;
    colsW.back      = colsW.back    + sumbac;      
    colsW.front     = colsW.front   + sumfro;
end
%_________________________________________________________________________%



% LEFT WALL OPERATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--Thermal Wall--%
%* Velocity Update
particle.vel(lef, 1) = leftW.dir * beta * sqrt(...
                        -2 * leftW.TW * log(rand(sumlef, 1)));
particle.vel(lef, 2) = beta * sqrt(leftW.TW) * randn(sumlef, 1);
particle.vel(lef, 3) = beta * sqrt(leftW.TW) * randn(sumlef, 1);
%* House-keeping
tof = dt * (particle.pos(lef, 1) - leftW.r) ./ ...
           (particle.pos(lef, 1) - particleOld(lef,1));
y_at_wall = particle.pos(lef, 2) - (tof / dt) .* ...
           (particle.pos(lef, 2) - particleOld(lef,2));
z_at_wall = particle.pos(lef, 3) - (tof / dt) .* ...
           (particle.pos(lef, 3) - particleOld(lef,3));
%* Position update
particle.pos(lef,1) = leftW.r + particle.vel(lef,1)...
                                 .* tof;
particle.pos(lef,2) = y_at_wall + particle.vel(lef,2)...
                                 .* tof;
particle.pos(lef,3) = z_at_wall + particle.vel(lef,3)...
                                 .* tof;
%_________________________________________________________________________%



% RIGHT WALL OPERATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%--Thermal Wall--%
%* Velocity Update
particle.vel(rig, 1) = rightW.dir * beta * sqrt(...
                        -2 * rightW.TW * log(rand(sumrig, 1)));
particle.vel(rig, 2) = beta * sqrt(rightW.TW)*randn(sumrig, 1);
particle.vel(rig, 3) = beta * sqrt(rightW.TW)*randn(sumrig, 1);
%* House-keeping
tof = dt * (particle.pos(rig, 1) - rightW.r) ./ ...
           (particle.pos(rig, 1) - particleOld(rig,1));
y_at_wall = particle.pos(rig, 2) - (tof / dt) .* ...
           (particle.pos(rig, 2) - particleOld(rig,2));
z_at_wall = particle.pos(rig, 3) - (tof / dt) .* ...
           (particle.pos(rig, 3) - particleOld(rig,3));
%* Position update
particle.pos(rig,1) = rightW.r + particle.vel(rig,1)...
                                 .* tof;
particle.pos(rig,2) = y_at_wall + particle.vel(rig,2)...
                                 .* tof;
particle.pos(rig,3) = z_at_wall + particle.vel(rig,3)...
                                .* tof;
%_________________________________________________________________________%



% BOTTOM WALL OPERATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%* Velocity Update
particle.vel(bot, 2) = bottomW.dir * beta * sqrt(...
                        -2 * bottomW.TW * log(rand(sumbot, 1)));
particle.vel(bot, 3) = beta*sqrt(bottomW.TW)*randn(sumbot, 1);
particle.vel(bot, 1) = beta*sqrt(bottomW.TW)*randn(sumbot, 1);
% House-keeping
tof = dt * (particle.pos(bot, 2) - bottomW.r) ./ ...
           (particle.pos(bot, 2) - particleOld(bot,2));
z_at_wall = particle.pos(bot, 3) - (tof / dt) .* ...
           (particle.pos(bot, 3) - particleOld(bot,3));
x_at_wall = particle.pos(bot, 1) - (tof / dt) .* ...
           (particle.pos(bot, 1) - particleOld(bot,1));
% Position update
particle.pos(bot,2) = bottomW.r + particle.vel(bot,2)...
                                 .* tof;
particle.pos(bot,3) = z_at_wall + particle.vel(bot,3)...
                                 .* tof;
particle.pos(bot,1) = x_at_wall + particle.vel(bot,1)...
                                 .* tof;
%_________________________________________________________________________%



% TOP WALL OPERATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-Thermal Wall--%
%* Velocity Update
particle.vel(top, 2) = topW.dir * beta * sqrt(...
                        -2 * topW.TW * log(rand(sumtop, 1)));
particle.vel(top, 3) = beta * sqrt(topW.TW) * randn(sumtop, 1);
particle.vel(top, 1) = beta * sqrt(topW.TW) * randn(sumtop, 1);
%* House-keeping
tof = dt * (particle.pos(top, 2) - topW.r) ./ ...
            (particle.pos(top, 2) - particleOld(top,2));
z_at_wall = particle.pos(top, 3) - (tof / dt) .* ...
            (particle.pos(top, 3) - particleOld(top,3));
x_at_wall = particle.pos(top, 1) - (tof / dt) .* ...
            (particle.pos(top, 1) - particleOld(top,1));
%* Position update
particle.pos(top,2) = topW.r + particle.vel(top,2)...
                                 .* tof;
particle.pos(top,3) = z_at_wall + particle.vel(top,3)...
                                 .* tof;
particle.pos(top,1) = x_at_wall + particle.vel(top,1)...
                                 .* tof;
%_________________________________________________________________________%



% FRONT WALL OPERATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%--Periodic Wall--%
particle.pos(fro, 3) = mod(particle.pos(fro, 3), L(3));
%_________________________________________________________________________%



% BACK WALL OPERATIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%--Periodic Wall--%
particle.pos(bac, 3) = mod(particle.pos(bac, 3), L(3));
%_________________________________________________________________________%


end
function collCELL(c)
% DSMC SOLVER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FILE:         collCELL.m
% AUTHOR:       Lachlan Schilling
% DATE (D/M/Y): 06/10/2022
% DESCRIPTION:  Collision function for the DSMC solver. 
%               
%_________________________________________________________________________%



% DEFINE GLOBALS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global debugMode coeffNTC colsP particle vRELMAX Nremcol;
%_________________________________________________________________________%



% FIND PARTICLES IN CELL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Using pre-sorted particle positions to find particle indices.
indxCELL = particle.id(c, 1) : (particle.id(c, 1) + particle.id(c, 2) - 1);
partCELL = particle.id(c, 2);
indxPARt = particle.cell(indxCELL, 2);
%_________________________________________________________________________%



% COLLIDE PARTICLES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if partCELL > 1 % omit cells with single particle
    
    % Extract the velocity of particles inside this cell
    vCELL = particle.vel(indxPARt, :);
    
    % Potential number of collisions
    truecols            = coeffNTC * partCELL * (partCELL - 1) * ...
                          vRELMAX(c) + Nremcol(c);  
    % Rounded off value
    cols        = floor(truecols);                                 
    Nremcol(c)	= truecols - cols;
    
    % loop over the potential number of collisions in this cell
    for icol = 1:cols
        
        % Pick any two particles at random
        p1 = ceil(rand * partCELL);
        p2 = ceil(rand * partCELL);
        
        % Calculate the realtive velocity between p1 and p2
        Vrel = sqrt((vCELL(p1,1) - vCELL(p2, 1))^2 + (vCELL(p1, 2) - ...
                vCELL(p2, 2))^2 + (vCELL(p1,3) - vCELL(p2, 3))^2);
        
        % Update the current maximum realtive velocity
        if Vrel > vRELMAX(c)
            vRELMAX(c) = Vrel;
        end
        
        % Collide only if the following criteria is true
        if (Vrel / vRELMAX(c)) > rand
            
            % Update the particle collision counter
            if debugMode
                colsP = colsP + 1;
            end
            
            % Center of mass frame of reference
            V_cm    = (vCELL(p1, :) + vCELL(p2, :)) / 2;    % COM velocity
            cosXi   = 2* rand - 1;                         
            sinXi   = sqrt(1 - cosXi^2);                    
            theta   = 2 * pi * rand;                     
            
            % Finding the post collision velocities
            Vrel_       = Vrel*[cosXi, sinXi*cos(theta), sinXi*sin(theta)];   
            vCELL(p1, :)= V_cm + Vrel_ / 2;                                         
            vCELL(p2, :)= V_cm - Vrel_ / 2;                                       
        end
    end
    
    % Update the post collision velocities---------------------
    particle.vel(indxPARt, :) = vCELL;
end
%_________________________________________________________________________%

end
function sampCELL()
% DSMC SOLVER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FILE:         sampCELL.m
% AUTHOR:       Lachlan Schilling
% DATE (D/M/Y): 07/10/2022
% DESCRIPTION:  Sampling function for the DSMC solver. Averages data over 
%               each cell for that timestep. When all 
%               simulations are completed, a movie is generated with the  
%               averaged data.
%_________________________________________________________________________%


% GLOBAL VARIABLES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global k m volCELL nPARTeff particle nCELL samples;
%_________________________________________________________________________%


% SAMPLE OVER ALL CELLS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for c = 1:nCELL
    
    % FIND PARTICLES IN CELL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Using pre-sorted particle positions to find particle indices.
    indxPART = particle.id(c,1) : (particle.id(c,1)+particle.id(c,2) - 1);
    partCELL = particle.id(c,2);
    indxPART = particle.cell(indxPART,2);
    %_____________________________________________________________________%
    
    
    % Extract the velocity of particles inside this cell %%%%%%%%%%%%%%%%%%
    vCELL = particle.vel(indxPART, :);
    %_____________________________________________________________________%
    
    
    % CELL BY CELL BASIS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    samples.vel(c,1:3) = sum(vCELL(:,1:3)) / partCELL;
    samples.V2(c,1:3)  = sum(vCELL(:,1).^2 + vCELL(:,2).^2 + vCELL(:,3).^2)...
        / partCELL; % Mean square of the velocities
    % Average temperature and Density in this cell
    % Note: The reaosn why we subtract the square mean velocities from the 
    % mean square velocity is to account for the bulk velocity of the
    % particles in each cell.
    samples.T(c)   = m / 3 / k * (samples.V2(c) - sumsqr(samples.vel(c)));
    samples.rho(c) = m * nPARTeff / volCELL * partCELL;
    %_____________________________________________________________________%
    
end
%_________________________________________________________________________%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
end
function saveDATA()
% DSMC SOLVER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FILE:         saveDATA.m
% AUTHOR:       Lachlan Schilling
% DATE (D/M/Y): 07/10/2022
% DESCRIPTION:  Save function for the DSMC solver. Averages data over 
%               the flow direction and saves each timestep.
%_________________________________________________________________________%


% GLOBAL VARIABLES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global debugMode colCELL colsP colsW samples printInfo save s steps;
%_________________________________________________________________________%


% average over flow direction z
for y = 1 : colCELL(2)
    for x = 1 : colCELL(1)
    	cZ = x + colCELL(1) * (y - 1) + ...
            colCELL(1) * colCELL(2) * (0 : colCELL(3) - 1);
        % properties at this time step in this simulation
        vx = sum(samples.vel(cZ, 1)) / colCELL(3);
        vy = sum(samples.vel(cZ, 2)) / colCELL(3);
        vt = sqrt(vx.^2 + vy.^2) / colCELL(3);
        T  = sum(samples.T(cZ))      / colCELL(3);
        rh = sum(samples.rho(cZ))    / colCELL(3);
        if s == 1
            save.vel(steps, x, y, 1) = vx;
            save.vel(steps, x, y, 2) = vy;
            save.vel(steps, x, y, 3) = vt;
            save.T(steps, x, y)      = T;
            save.rho(steps, x, y)    = rh;
        else % average the data on the fly
            save.vel(steps, x, y, 1) = save.vel(steps, x, y, 1)*...
                (s - 1) / s + vx * (1 / s);
            save.vel(steps, x, y, 2) = save.vel(steps, x, y, 2)*...
                (s - 1) / s + vy * (1 / s);
            save.vel(steps, x, y, 3) = save.vel(steps, x, y, 3)*...
                (s - 1) / s + vt * (1 / s);
            save.T(steps, x, y)      = save.T(steps, x, y)*...
                (s - 1) / s + T  * (1 / s);
            save.rho(steps, x, y)    = save.rho(steps, x, y)*...
                (s - 1) / s + rh * (1 / s);
        end
    end
end


% print information to the user if a boolean if true
if printInfo
    if s < 10
        st = "00" + num2str(s);
    elseif s < 100
        st = "0" + num2str(s);
    else
        st = num2str(s);
    end
    if steps == 1
        fprintf('------- SIMULATION #%s -------\n', st);
    end
    if mod(steps, 10) == 0
        fprintf('Step:\t%i\n',steps);
    end
    if mod(steps,10) == 0 && debugMode
        fprintf('Wall Collisions\n');
        fprintf('\tLeft:\t%i\n\tRight:\t%i\n\tTop:\t%i\n\tBottom:\t%i\n',...
            colsW.left, colsW.right, colsW.top, colsW.bottom);
        fprintf('Particle Collisions\n\tcolsP:\t%i\n', colsP);
        fprintf('\n');
    end
end  
	
end
function doMovie()
% DSMC SOLVER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% FILE:         doMovie.m
% AUTHOR:       Lachlan Schilling
% DATE (D/M/Y): 07/10/2022
% DESCRIPTION:  Movie function for the DSMC solver. Creates movies from the
%               save data over every timestep.
%_________________________________________________________________________%



% GLOBAL VARIABLES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global T0 TW L colCELL N_dt save attempt doTiles;
%_________________________________________________________________________%



% OPEN MOVIE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if attempt < 10
	st = "00" + num2str(attempt);
elseif attempt < 100
	st = "0" + num2str(attempt);
end
v = VideoWriter('attempt' + st, 'MPEG-4');
v.Quality = 100;
v.FrameRate = 150;
open(v);
%_________________________________________________________________________%



% FRAMES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if doTiles
    t = tiledlayout(1,2);
    a = nexttile;
end
x = (0:colCELL(1)) / colCELL(1);
y = (0:colCELL(2)) / colCELL(2);
[XXX, YYY]  = meshgrid(x, y);
levels      =  T0*0.9:(TW-T0)/50:TW; % 0.9
inputT      = TW*ones(colCELL(1) + 1, colCELL(2) + 1);
if ~doTiles
    h = figure;
end
hold on
g = colorbar;
g.Title.String = 'Temperature (K)';
caxis([T0 TW]);
colormap jet
xlim([0 1])
ylim([0 1])
axis equal
Told = T0;
if L(1) == 1e-3
    xlabel('X (mm)')
    ylabel('Y (mm)')
elseif L(1) == 1e-6
    xlabel('X (µm)')
    ylabel('Y (µm)')
end
if doTiles
    b = nexttile;
    xlabel('Time Steps')
    ylabel('Mean Temperature (K)')
    xlim([0 N_dt])
    ylim([T0 TW]);
    axes(a)
end
for i = 1:N_dt
    
    % steps
    if i < 10
        istring = "00" + num2str(i);
    elseif i < 100
        istring = "0" + num2str(i);
    else
        istring = num2str(i);
    end
    
    % temperature array
    for jj = 2:colCELL(2)
        for ii = 2:colCELL(1)
            inputT(ii, jj) = 0.25 * (save.T(i, ii - 1, jj -1) ...
                                   + save.T(i, ii, jj -1)...
                                   + save.T(i, ii, jj)...
                                   + save.T(i, ii - 1, jj));
        end
    end
    
    % contour
    [~,hot] = contourf(XXX,YYY,inputT,levels);
    title("Temperature Contours: step = " + istring)
    hot.EdgeColor = 'none';
    
    % tile stuff
    if doTiles
        axes(b);
        hold on
        Tav = mean(save.T(i));
        plot([i - 1, i], [Told, Tav], 'r')
        axes(a);
        Told = Tav;
    end
    
    % video stuff
    frame = getframe(h);
    writeVideo(v, frame);
    cla
end
hold off
close(v);
close all;
%_________________________________________________________________________%
end
function tiledPlot()

    % globals
    global T0 TW N_dt save;

    % plotting
    T_av = mean(save.T,[2 3]);
    plot(1:N_dt, T_av, 'b')
        xlabel('Time Steps')
    ylabel('Mean Temperature (K)')
    xlim([0 N_dt])
    ylim([T0 TW]);

end
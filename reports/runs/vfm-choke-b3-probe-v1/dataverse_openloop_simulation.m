clear
clc

%--------------------------------------------------------------------------
%% Settings
%--------------------------------------------------------------------------
    timespan = 100; %[hr]
    dt = 20; %[s]
    %----------------------------------------------------------------------
    % Well    	   = [Well1    Well2    Well1     Well2     Well1    Well2]
    % PAR    	   = [    --PI--            --GOR--             --WC--    ]
    % UNCERTAINTY  = [ PI1      PI2      GOR1      GOR2      WC1      WC2 ]
    Theta_nom      = [2.51      1.63     0.08      0.07      0.15     0.15];
    
    %----------------------------------------------------------------------
    % Select the type of input disturbance, w_gc
    % NOTE: 1st column of w_gc correspond to value of w_gc in Sm3/hr
    %       2nd column of w_gc correspond to ratio of step lengths
%     type = 'constant';    w_gc_val = [  40000, 1];
%     type = '1step';       w_gc_val = [  40000, 1;
%                                         36000, 2];
    type = '2step';       w_gc_val = [  40000, 1;
                                        20000, 1;
                                        40000, 1];
    %----------------------------------------------------------------------
    % Specify the percentage distribution of w_gc to each well
    % NOTE: Sum should be <= 100%
    % Well    	= [Well1    Well2]
    w_gc_dist 	= [30   	30]; %[%]

%--------------------------------------------------------------------------
%% Confirming the settings
%--------------------------------------------------------------------------
    if sum(w_gc_dist) > 100 % if % distribution of w_gc to each well adds up to more than 100%
        error('Percentage distribution of w_gc to each well cannot add up to more than 100%')
    end

    % Displaying the settings
    fprintf('OPENLOOP SIMULATION SETTINGS \n')
    fprintf('	dt          : %g s                  \n', dt)
    fprintf('	timespan	: %g hrs                \n', timespan)
    fprintf('	w_gc        : %s, [%s] Sm3/hr       \n', type, join(string(w_gc_val(:,1))," "))
    fprintf('	w_ga        : [%s]%%*w_gc           \n', join(string(round(w_gc_dist,1))," "))
    fprintf('	Theta_nom  : [%s]  \n', join(string(Theta_nom)," "))

    % Prompting the user to confirm the settings
%     disp(' ')
%     disp('Please make sure the above settings are okay. Then press any key to start the simulation OR press ctrl + c to stop execution.')
%     pause

%--------------------------------------------------------------------------
%% Creating other necessary data
%--------------------------------------------------------------------------
    t = 0:dt:timespan*60*60; %[s]

    % Well     	= [Well1    Well2]
    m_ga    	= [16422   	15445]; %[kg]
    m_gt      	= [1669    	1517]; %[kg]
    m_ot      	= [15229   	13458]; %[kg]
    m0 = [m_ga, m_gt, m_ot];

    w_gc = make_w_gc(type, w_gc_val, t); %[kg/s]

    w_ga = w_gc_dist/100.*w_gc; %[kg/s]

%--------------------------------------------------------------------------
%% Running openloop simulation
%--------------------------------------------------------------------------
    % Initializing the ODE solver
    update_states = @(m0_k, w_ga_k)update_states(m0_k, w_ga_k, Theta_nom, dt);

    % Preallocation
    m = zeros(length(t),length(m0));    m(1,:) = m0;
    Theta_OL  	 = Theta_nom.*ones(length(t),1);

    % For each time step within the simulation timespan
    for k = 1:length(t)-1
        m(k+1,:) = update_states(m(k,:), w_ga(k,:));
    end

    [w_op, w_gop] = calculate_output(m, Theta_OL);

%--------------------------------------------------------------------------
%% Plotting the results
%--------------------------------------------------------------------------
    timestamps = t/60/60; %[hrs]
    x_label = 'Time [hr]';
    legend_labels = {'well 1','well 2'};

    figure
    tiledlayout(2,1,'Padding','compact', 'TileSpacing','compact')
    nexttile;
    plot(timestamps,sum(w_op,2))
    title('Total oil production rate, \Sigma w_{op}^i');
%     ylim('padded'); xlim('tight')
    ylims = ylim; x = diff(ylim)*0.1; ylim([ylims(1)-x, ylims(2)+x])
    ylabel('[kg/s]')
    grid on
    xlabel('time [hrs]')

    nexttile
    plot(timestamps,sum(w_gop,2));
    title('Total fluid production rate \Sigma w_{gop}^i');
    xlim([timestamps(1), timestamps(end)]);
%     yline(160,'--r')
    ylabel('[kg/s]');
    xlabel('time [hrs]')
    grid on

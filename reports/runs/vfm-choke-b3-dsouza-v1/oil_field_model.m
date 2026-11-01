function  [dm, w_op, w_gop, P_a, P_ainj, rho_l, V_G, rho_m, P_tinj, Y_2, rho_ga, w_ginj, P_wf, w_l, w_o, w_g, P_wh, Y_3, w_gp, w_lp, w_wp] = oil_field_model(m, w_ga, Theta)

% NOTE:
%     * This function can accept matrix inputs. But make sure rows of m,
%       w_ga & d_PI correspond to instances.
%
%     * Outputs are row vectors or, for matrix inputs, their instances are
%       arranged in rows.

%--------------------------------------------------------------------------
%% Extracting info. from the inputs
%--------------------------------------------------------------------------
    % Finding the number of instances
    no_inst = size(m,1);

    % Extracting the 3 states separately from m
    no_wells = 2;
    m_ga  = m(:, 1            :no_wells);
    m_gt  = m(:, no_wells+1   :2*no_wells);
    m_lt  = m(:, 2*no_wells+1 :end);
    
    PI  = Theta(:,1:2);
    GOR = Theta(:,3:4);
    WC  = Theta(:,5:6);

%--------------------------------------------------------------------------
%% Parameters
%--------------------------------------------------------------------------
    % Well      = [Well1    Well2] 
    L_a_tl      = [2758     2559].*ones(no_inst,1); %[m]
    L_t_tl      = L_a_tl; %[m]
    L_a_vl      = [2271     2344].*ones(no_inst,1); %[m]
    L_t_vl      = L_a_vl; %[m]
    ID_t        = [6.18     6.18]*0.0254.*ones(no_inst,1); %[m]
    A_t         = pi/4*ID_t.^2;
    ID_a        = [9.63     9.63]*0.0254.*ones(no_inst,1); %[m]
    OD_t        = [7.64     7.64]*0.0254.*ones(no_inst,1); %[m]
    A_a      	= pi/4*(ID_a.^2 - OD_t.^2);
    L_r_vl      = [114      67].*ones(no_inst,1); %[m]
    K           = [68.43    67.82].*ones(no_inst,1); %[sqrt(kg*m3/bar)/hr]
    T_a        	= [280      280].*ones(no_inst,1); %[K]
    T_t        	= [280      280].*ones(no_inst,1); %[K]
    u_2         = [100      100].*ones(no_inst,1); %[%]

    P_r         = 150; %[bar]
    P_s         = 30; %[bar]
    N_6         = 27.3;
    alpha_Y     = 0.66;
    rho_o       = 800; %[kg/m3]
    rho_w       = 1000; %[kg/m3]
    M          	= 20*1e-3; %[kg/mol]
    z           = 1.3;
    P_ainj_min  = 0; %[bar]
    P_wh_min    = 0; %[bar]
    R           = 8.31446261815324; %[J/K/mol]
    g           = 9.80665; %[m/s2]

%--------------------------------------------------------------------------
%% Algebraic equations
%--------------------------------------------------------------------------
    P_a = z*m_ga*R.*T_a./(M*A_a.*L_a_tl)*1e-5; %[bar] (6)

    P_ainj = P_a + m_ga*g.*L_a_vl./(A_a.*L_a_tl)*1e-5; %[bar] (11)
    
    rho_l = rho_w.*WC + rho_o.*(1-WC);

    V_G = A_t.*L_t_tl - m_lt./rho_l; %[m3] (eq. after 13)
    
    rho_m = (m_gt + m_lt)./(A_t.*L_t_tl); %[kg/m3] (eq. after 15)
    
    P_tinj = (z*m_gt*R.*T_t./(M*V_G) + rho_m*g.*L_t_vl/2)*1e-5; %[bar] (17)
    
    Y_2 = 1 - alpha_Y*(P_ainj - P_tinj)./max(P_ainj, P_ainj_min); %[const.] (eq. after 10)
    
    rho_ga = M*(P_a + P_ainj)*1e5./(2*z*R*T_a); %[kg/m3] (12)

    w_ginj = K.*Y_2.*sqrt(rho_ga.*max(P_ainj - P_tinj, 0))/3600; %[kg/s] (10)
    %----------------------------------------------------------------------
    P_wf = P_tinj + rho_l.*g.*L_r_vl*1e-5; %[bar] (18)
    
    w_l = PI.*max(P_r - P_wf, 0); %[kg/s] (19)
    
    w_o = (rho_o./rho_l).*(1-WC).*w_l;

    w_g = GOR.*w_o; %[kg/s] (19)
    %----------------------------------------------------------------------
    P_wh = (z*m_gt*R.*T_t./(M*V_G) - rho_m*g.*L_t_vl/2)*1e-5; %[bar] (16)
    
    Y_3 = 1 - alpha_Y*(P_wh - P_s)./max(P_wh, P_wh_min); %[const.] (eq. after 20)
    
    w_gop = 10*N_6*(0.5*u_2 - 20).*Y_3.*sqrt(rho_m.*max(P_wh - P_s, 0))/3600; %[kg/s] (20)
    %----------------------------------------------------------------------
    w_gp = m_gt./(m_gt + m_lt).*w_gop; %[kg/s] (21)
    %----------------------------------------------------------------------
    w_lp = m_lt./(m_gt + m_lt).*w_gop; %[kg/s] (22)
    
    w_op = (rho_o./rho_l).*(1-WC).*w_lp;
    
    w_wp = w_lp - w_op;

%--------------------------------------------------------------------------
%% Differential equations
%--------------------------------------------------------------------------
    dm_ga = w_ga - w_ginj; %[kg/s] (9)
    dm_gt = w_ginj + w_g - w_gp; %[kg/s] (23)
    dm_lt = w_l - w_lp; %[kg/s] (24)

    dm = [dm_ga, dm_gt, dm_lt];
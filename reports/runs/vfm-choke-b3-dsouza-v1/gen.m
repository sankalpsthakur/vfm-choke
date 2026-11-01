
Theta_nom = [2.51 1.63 0.08 0.07 0.15 0.15];
dt = 20; timespan = 96;
t = 0:dt:timespan*60*60;
w_gc = make_w_gc('2step', [40000 1; 20000 1; 40000 1], t);
w_ga = [30 30]/100 .* w_gc;
m0 = [16422 15445 1669 1517 15229 13458];
m = zeros(length(t), 6); m(1,:) = m0;
for k = 1:length(t)-1
  m(k+1,:) = update_states(m(k,:), w_ga(k,:), Theta_nom, dt);
end
Theta_OL = Theta_nom .* ones(length(t),1);
[dm, w_op, w_gop, P_a, P_ainj, rho_l, V_G, rho_m, P_tinj, Y_2, rho_ga, w_ginj, P_wf, w_l, w_o, w_g, P_wh, Y_3, w_gp, w_lp, w_wp] = oil_field_model(m, zeros(length(t),2), Theta_OL);
w_ga_full = [w_ga; w_ga(end,:)];
for wi = 1:2
  M = [t'/3600, w_ga_full(:,wi), P_wf(:,wi), P_wh(:,wi), w_op(:,wi), w_gp(:,wi), w_lp(:,wi)];
  dlmwrite(sprintf('raw_well%d.csv', wi), M, 'precision', 12);
end

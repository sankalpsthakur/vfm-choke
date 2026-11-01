function m_next = update_states(m0, w_ga, Theta, dt)

% RK4 algorithm

dm1 = oil_field_model(m0, w_ga, Theta);
dm2 = oil_field_model(m0+0.5*dt*dm1, w_ga, Theta);
dm3 = oil_field_model(m0+0.5*dt*dm2, w_ga, Theta);
dm4 = oil_field_model(m0+dt*dm3, w_ga, Theta);

m_next = m0 + dt/6*(dm1 + 2*dm2 + 2*dm3 + dm4);

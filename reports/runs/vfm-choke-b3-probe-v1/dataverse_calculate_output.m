function [w_op, w_gop] = calculate_output(m, Theta)

% NOTE
%     * w_ga & d_PI is not required for calculating w_op & w_gop using the
%       oil_field_model; they are required only for calculating the
%       derivatives of the states. So, this function passes a dummy_w_ga &
%       a dummy_d_PI to oil_field_model.

dummy_w_ga = zeros(size(m,1),2);
% dummy_d_PI = zeros(size(m,1),2);

[~, w_op, w_gop] = oil_field_model(m, dummy_w_ga, Theta);

function w_gc = make_w_gc(type, w_gc_val_rat, t)

% Length of w_gc
len = length(t) - 1;

% Extract w_gc values and step ratio separately
w_gc_val    = w_gc_val_rat(:,1);
step_ratio  = w_gc_val_rat(:,2);

% Change the units of input disturbance values
w_gc_val = w_gc_val*0.83/60/60; %[kg/s]

switch type
    case 'constant'
        w_gc = w_gc_val*ones(len,1); %[kg/s]

    case '1step'
        % Lengths of the steps
        step_len1 = floor(len*step_ratio(1)/sum(step_ratio));
        step_len2 = len - step_len1;

        % Input disturbance
        w_gc = [w_gc_val(1)*ones(step_len1,1);
                w_gc_val(2)*ones(step_len2,1)]; %[kg/s]

    case '2step'
        % Lengths of the steps
        step_len1 = floor(len*step_ratio(1)/sum(step_ratio));
        step_len2 = floor(len*step_ratio(2)/sum(step_ratio));
        step_len3 = len - step_len1 - step_len2;

        % Input disturbance
        w_gc = [w_gc_val(1)*ones(step_len1,1);
                w_gc_val(2)*ones(step_len2,1);
                w_gc_val(3)*ones(step_len3,1)]; %[kg/s]
end

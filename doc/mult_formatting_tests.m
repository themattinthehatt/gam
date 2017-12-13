% mult_formatting_tests.m

%% create data

% a1 <- g1
% a2 <- g1, g2
% a3 <- g2, g3
% a4 <- g3
% a5 <- g2, g4
%
% only g1 and g3 will be fit, so that
%
% a1*g1 + a2*g1*g2 + a3*g2*g3 + a4*g3 + a5*g2*g4
% =>
% g1*(a1 + a2*g2) + 
% g3*(a3*g2 + a4) +
% a5*g2*g4

T = 100;
num_neurons = 4;

num_add_subunits = 5;
num_mult_subunits = 4;
add_targets = {[1, 2], ...
               [2, 3, 5], ...
               [3, 4], ...
               5};

input_params = GAM.create_input_params([1, num_neurons, 1]);
for i = 1:max(num_add_subunits, num_mult_subunits)
    Xstims{i} = GAM.create_time_embedding(i*ones(T,num_neurons), input_params);
end

%% initialize models - 5 add, 4 mult
  
% initialize net
net0 = GAM( ...
            'noise_dist', 'gauss', ... 
            'spiking_nl', 'lin');

% add add_subunits
for i = 1:num_add_subunits
    net0 = net0.create_subunit('add', num_neurons, [], ...
            i, input_params, ...
            'act_funcs', 'lin');
    net0.add_subunits(i).layers.weights = eye(num_neurons);
    net0.add_subunits(i).layers.biases = zeros(num_neurons,1);
end

% add mult_subunits
for i = 1:num_mult_subunits
    net0 = net0.create_subunit('mult', num_neurons, add_targets{i}, ...
            i, input_params, ...
            'act_funcs', 'lin');
    net0.mult_subunits(i).layers.weights = eye(num_neurons);
    net0.mult_subunits(i).layers.biases = zeros(num_neurons,1);
end

% build fit_struct as in fit_model
fit_struct.fit_type = 'mult';
fit_struct.true_act = NaN;
fit_struct.fit_subs_add = [];
fit_struct.fit_subs_mult = [1, 3];
fit_struct.Xstims = cellfun(@(x) x', Xstims, 'UniformOutput', 0);

% format model for add_subunits fitting
[net1, fit_struct1] = net0.format_for_mult_fit(fit_struct);

% get internals
[~,~,~,add_subs_out, mult_subs_comb, mult_subs_out] = ...
    net1.get_model_internals(cellfun(@(x) x', fit_struct1.Xstims, ...
                                    'UniformOutput', 0));

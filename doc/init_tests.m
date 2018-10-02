% init_tests.m

%% create data

% create simulated population data that contains both additive and 
% multiplicative terms
createSimData;

% fit normalized 2p data
data = data_struct.data_fr;

num_neurons = data_struct.meta.num_neurons;

%% model-fitting options

deriv_check = 0;
display = 'iter';

model_arch.oneadd_onemult = 1;
model_arch.oneadd = 0;
model_arch.twoadd = 0;
model_arch.twoadd_onemult1 = 0;
model_arch.twoadd_onemult2 = 0;
model_arch.twoadd_twomult12 = 0;

%% create input matrix for additive subunit

% specify input parameters
num_lags = 1;
tent_spacing = 1;
input_params(1) = GAM.create_input_params( ...
            [num_lags, num_neurons, 1], ... 
            'tent_spacing', tent_spacing);
       
% create lagged input matrix
Xstims{1} = GAM.create_time_embedding(data, input_params(1));

%% create input matrix for additive subunit

rand_dim = 4;

% specify input parameters
num_lags = 1;
tent_spacing = 1;
input_params(2) = GAM.create_input_params( ...
            [num_lags, rand_dim, 1], ... 
            'tent_spacing', tent_spacing);
       
% create lagged input matrix
Xstims{2} = GAM.create_time_embedding(data*randn(num_neurons, rand_dim), ...
                                      input_params(2));

%% initialize models - one add, one mult

if model_arch.oneadd_onemult
    
    % initialize net
    net0 = GAM( ...
                'noise_dist', 'gauss', ... 
                'spiking_nl', 'lin');

    % add add_subunit
    net0 = net0.create_subunit('add', [3, num_neurons], [], ...
                1, input_params(1), ...
                'act_funcs', {'lin', 'relu'});

    % add mult_subunit
    net0 = net0.create_subunit('mult', [1, num_neurons], 1, ...
                1, input_params(1), ...
                'act_funcs', {'relu', 'oneplus'}, ...
                'init_params', 'gauss');

    % update model params
    net0 = net0.set_reg_params( ...
                'all', ...
                'l2_weights', 1e-4, ...
                'l2_biases', 1e-6);
    net0 = net0.set_optim_params( ...
                'max_iter', 5000, ...
                'display', display, ...
                'deriv_check', 0);
            
    % fit add subunits
    net0 = net0.fit_model('add_subunits', data, Xstims);

    % evaluate model
    [mod_meas0, mod_int0, mod_reg_pen0] = net0.get_model_eval(data, Xstims);

    % fit mult subunits
    net1 = net0.fit_model('mult_subunits', data, Xstims);
    
    % evaluate model
    [mod_meas1, mod_int1, mod_reg_pen1] = net1.get_model_eval(data, Xstims);
    
end
                                         
%%

[a, z] = net1.add_subunits.get_model_internals(Xstims);

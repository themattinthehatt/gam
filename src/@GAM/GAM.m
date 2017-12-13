classdef GAM
    
% Class implementing a generalized affine model in which the activity of a 
% population of neurons is modeled using a stimulus model for each neuron,
% modulated by a multiplicative gain term that is shared across the
% population
%
% Author: Matt Whiteway
%   08/31/17

properties
    add_subunits            % array of additive subunits
    mult_subunits           % array of multiplicative subunits
    noise_dist              % noise distribution for output
    spiking_nl              % final nonlinearity on output
    num_neurons             % number of simultaneous neurons to fit
    biases                  % bias terms added before spiking_nl
    optim_params            % struct of optimization parameters
        % opt_routine
        % max_iter
        % display
        % monitor
        % deriv_check
    fit_history             % struct array; saves result of each fit step
end

properties (Hidden)
    version = '1.0';        % version number
    date = date;            % date of fit
    min_pred_rate = 1e-5;   % min val for logs
    max_g = 50;             % max val for exponentials
    
    % user options
    allowed_spiking_nls = ...
        {'lin', ...         % f(x) = x
         'relu', ...        % f(x) = max(0, x)
         'sigmoid', ...     % f(x) = 1 / (1 + exp(-x))
         'softplus', ...    % f(x) = log(1 + exp(x))
         'exp', ...         % f(x) = exp(x)
        };
    allowed_noise_dists = ...
        {'gauss', ...
        'poiss'};
    allowed_optimizers = ...
        {'fminunc', ...
         'minFunc', ...
         'con'};
end

% methods that are implemented in separate files
methods
    net = fit_add_subunits(net, fit_struct);
    net = fit_add_subunits_op(net, fit_struct);
end


%% ********************* constructor **************************************
methods
      
    function net = GAM(varargin)
    % net = GAM(kargs) 
    %
    % Constructor function for an GAM object; sets properties, but does not
    % build any model subunits. To add subunits, the add_subunt function 
    % must be called after initializing the GAM object.
    %
    % INPUTS:
    %
    %   optional key-value pairs: [defaults]
    %       'noise_dist', string
    %           ['gauss'] | 'poiss'
    %           specifies noise distribution for cost function
    %       'spiking_nl', string
    %           ['lin'] | 'relu' | 'softplus' | 'sigmoid' | 'exp'
    %           output nonlinearity for each cell
    %       'num_neurons', scalar
    %           defaults to empty vector; will be updated once first
    %           subunit is added           
    %
    % OUTPUT:
    %   net: initialized GAM object
    
    if nargin == 0
        % handle the no-input-argument case by returning a null model. This
        % is important when initializing arrays of objects
        return 
    end

    % define defaults    
    noise_dist = 'gauss';
    spiking_nl = 'lin';
    num_neurons = [];
   
    % parse varargin
    assert(mod(length(varargin), 2) == 0, ...
        'GAM:Constructor:Input should be a list of key-value pairs')
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'noise_dist'
                assert(ismember(varargin{i+1}, net.allowed_noise_dists),...
                    'GAM:Constructor:Invalid noise distribution "%s"', ...
                    varargin{i+1})
                noise_dist = varargin{i+1};
            case 'spiking_nl'
                assert(ismember(varargin{i+1}, net.allowed_spiking_nls), ...
                    'GAM:Constructor:Invalid spiking_nl "%s"', varargin{i+1})
                spiking_nl = varargin{i+1};
            case 'num_neurons'
                assert(isinteger(varargin{i+1}), ...
                    'GAM:Constructor:num_neurons must be an integer')
                assert(varargin{i+1} > 0, ...
                    'GAM:Constructor:num_neurons must be greater than zero')
                num_neurons = varargin{i+1};
            otherwise
                error('GAM:Constructor:Invalid input flag "%s"', ...
                    varargin{i});
        end
        i = i + 2;
    end
   
    % set properties
    net.add_subunits = [];
    net.mult_subunits = [];
    net.noise_dist = noise_dist;
    net.spiking_nl = spiking_nl;
    net.num_neurons = num_neurons;
    if isempty(net.num_neurons)
        net.biases = [];
    else
        net.biases = zeros(net.num_neurons, 1);
    end
    net.optim_params = GAM.set_init_optim_params();
    net.fit_history = struct([]);    
    
    end % method

end

%% ********************* setting methods **********************************
methods
    
    function net = set_optim_params(net, varargin)
    % net = net.set_optim_params(kargs)
    %
    % Takes a sequence of key-value pairs to set optimization parameters 
    % for an GAM object
    %
    % INPUTS:
    %   optional key-value pairs:
    %       'optimizer', string     
    %           specify the optimization routine used for learning the
    %           weights and biases; see allowed_optimizers for options
    %       'display', string
    %           'off' | 'iter'
    %           'off' to suppress output, 'iter' for output at each
    %           iteration
    %       'max_iter', scalar
    %           maximum number of iterations of optimization routine
    %       'monitor', string
    %           'off' | 'iter'
    %           'off' to suppress saving output, 'iter' to save output 
    %           at each iteration
    %       'deriv_check', boolean
    %           specifies numerical derivative checking
    %
    % OUTPUTS:
    %   net: updated GAM object
    
    % parse varargin
    assert(mod(length(varargin), 2) == 0, ...
        'GAM:set_optim_params:Input should be a list of key-value pairs')
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'optimizer'
                assert(ismember(varargin{i+1}, net.allowed_optimizers), ...
                    'GAM:set_optim_params:Invalid optimizer "%s"', ...
                    varargin{i+1})
                net.optim_params.optimizer = varargin{i+1};
            case 'display'
                assert(ismember(varargin{i+1}, {'off', 'iter'}), ...
                    'GAM:set_optim_params:Invalid display option "%s"', ...
                    varargin{i+1})
                net.optim_params.Display = varargin{i+1};
            case 'max_iter'
                assert(varargin{i+1} > 0, ...
                    ['GAM:set_optim_params:', ...
                    'Max number of iterations must be greater than zero'])
                net.optim_params.max_iter = varargin{i+1};      % internal
                net.optim_params.maxIter = varargin{i+1};       % minFunc and fminunc
                net.optim_params.maxFunEvals = 2*varargin{i+1};
            case 'monitor'
                assert(ismember(varargin{i+1}, {'off', 'iter'}), ...
                    'GAM:set_optim_params:Invalid monitor option "%s"', ...
                    varargin{i+1})
                net.optim_params.monitor = varargin{i+1};
            case 'deriv_check'
                assert(ismember(varargin{i+1}, [0, 1]), ...
                    ['GAM:set_optim_params:', ...
                    'deriv_check option must be set to 0 or 1'])
                net.optim_params.deriv_check = varargin{i+1};
            otherwise
                error('GAM:set_optim_params:Invalid input flag "%s"', ...
                    varargin{i});
        end
        i = i + 2;
    end
    
    end % method
    
    
    function net = set_reg_params(net, reg_target, varargin)
    % net = net.set_reg_params(kargs)
    %
    % Takes a sequence of key-value pairs to set regularization parameters
    % for subunits in GAM object. 
    %
    % example: set reg params for all subunits
    %   net = net.set_reg_params('all', 'l2_weights', 10, 'l2_biases', 1)
    % example: set reg params for all additive subunits
    %   net = net.set_reg_params('add', 'l2_weights', 10, 'l2_biases', 1) 
    % example: set reg params for specific multiplicative subunits
    %   net = net.set_reg_params('mult', 'subs', [1, 3], 'l2_biases', 1) 
    %
    % Note 1: both the Layer and Subunit classes have an equivalent method; 
    % the main usefulness of this method is to quickly update the
    % reg_params structure for ALL subunits and layers
    % Note 2: this method updates all layers of the specified subunits with
    % a single value; to update layers of a single subunit individually,
    % the Subunit.set_reg_params function will need to be called separately
    % for each layer
    %
    % INPUTS:
    %   reg_target:     
    %       'all' | 'add' | 'mult'
    %       string specifying which model components to apply the specified 
    %       reg params to
    %
    %   optional key-value pairs:
    %       'subs', vector
    %           specify set of subunits to apply the new reg_params
    %           to (default: all subunits of specified type)
    %       'reg_type', scalar
    %           'l2_weights' | 'l2_biases'
    %           first input is a string specifying the type of 
    %           regularization, followed by a scalar giving the associated 
    %           regularization value, which will be applied to the subunits
    %           specified by 'subs'
    %   
    % OUTPUTS:
    %   net: updated GAM object
    
    % error-check inputs
    assert(ismember(reg_target, {'all', 'add', 'mult'}), ...
        'GAM:set_reg_params:Invalid subunit type "%s"', reg_target)
    
    % define defaults
    add_targets = 1:length(net.add_subunits);
    mult_targets = 1:length(net.mult_subunits);
    
    % pull out 'subs' if present
    subs_loc = find(strcmp(varargin, 'subs'));
    if ~isempty(subs_loc)
        
        % verify subs_loc is valid
        if any(strcmp(reg_target, {'all', 'add'}))
            assert(all(ismember(varargin{subs_loc+1}, ...
                                1:length(net.add_subunits))), ...
                'GAM:set_reg_params:Invalid add subunits specified')
            add_targets = varargin{subs_loc+1};
        end
        if any(strcmp(reg_target, {'all', 'mult'}))
            assert(all(ismember(varargin{subs_loc+1}, ...
                                1:length(net.mult_subunits))), ...
                'GAM:set_reg_params:Invalid mult subunits specified')
            mult_targets = varargin{subs_loc+1};
        end
                
        % remove 'subs' from varargin; will be passed to another method
        varargin(subs_loc) = [];
        varargin(subs_loc) = []; % equiv to subs_loc+1 after deletion
        
    end
    
    % update regs
    if any(strcmp(reg_target, {'all', 'add'}))
        for i = add_targets
            net.add_subunits(i) = ...
                net.add_subunits(i).set_reg_params(varargin{:}); 
        end
    end
    if any(strcmp(reg_target, {'all', 'mult'}))
        for i = mult_targets
            net.mult_subunits(i) = ...
                net.mult_subunits(i).set_reg_params(varargin{:}); 
        end
    end

    end % method
	
    
    function net = create_subunit(net, sub_type, layer_sizes, add_targets, ...
                                  input_target, input_params, varargin)
    % net = net.create_subunit(sub_type, layer_sizes, add_targets, ...
    %                          input_target, input_params, kargs)
    %
    % Creates a new GAMSubunit object and adds it to current GAM object
    %
    % INPUTS:
    %   sub_type:       'add' | 'mult'
    %                   string specifying type of subunit to create
    %   layer_sizes:    scalar array of number of nodes in each layer,
    %                   excluding the input layer
    %   add_targets:    if creating add_subunit: empty
    %                   if creating mult_subunit: list of additive targets
    %                   that this subunit will multiply
    %   input_target:   scalar that specifies which input matrix to use
    %   input_params:   struct that defines params of input matrix
    %
    %   optional key-value pairs: [defaults]
    %       'act_funcs', cell array of strings, one for each (non-input)
    %           layer of the subunit
    %           ['lin'] | 'relu' | 'sigmoid' | 'softplus' | 'oneplus' |
    %           'exp'
    %           If all layers share the same act_func, use a single string
    %       'init_params', cell array of strings, one for each (non-input)
    %           layer of the subunit
    %           ['gauss'] | 'trunc_gauss' | 'uniform' | 'orth' | 'zeros'
    %           If all layers share the same init_params, use a single
    %           string
    %
    % OUTPUT:
    %   subunit: updated GAM object

    % most error-checking on inputs performed in GAMSubunit constructor
    if ~isempty(net.num_neurons)
        assert(layer_sizes(end) == net.num_neurons, ...
            'GAM:create_subunt:Output layer does not match num_neurons')
    else
        net.num_neurons = layer_sizes(end);
        net.biases = zeros(net.num_neurons, 1);
    end
    
    % add new subunit to GAM object
    switch sub_type
        case 'add'
            if isempty(net.add_subunits)
                net.add_subunits = GAMSubunit(layer_sizes, [], [], ...
                                            input_target, input_params, ...
                                            varargin{:});
            else
                net.add_subunits(end+1,1) = GAMSubunit(layer_sizes, [], [], ...
                                            input_target, input_params, ...
                                            varargin{:});
            end
        case 'mult'
            if isempty(net.mult_subunits)
                net.mult_subunits = GAMSubunit(layer_sizes, ...
                                            add_targets, [], ...
                                            input_target, input_params, ...
                                            varargin{:});
            else
                net.mult_subunits(end+1,1) = GAMSubunit(layer_sizes, ...
                                            add_targets, [], ...
                                            input_target, input_params, ...
                                            varargin{:});
            end
        otherwise
            error('GAM:create_subunit:Invalid subunit type "%s"', sub_type)
    end
    
    % check to make sure all additive and multiplicative subunits are
    % compatible
    for i = 1:length(net.mult_subunits)
        add_targets = net.mult_subunits(i).add_targets;
        for j = add_targets
            % check for existence of add_subunit
            if length(net.add_subunits) < j
                % throw warning; subunit might be added later
                warning(['GAM:create_subunit:', ...
                    'targeted add_subunit does not exist!'])
            end
            % check for existence of current mult_subunit in list
            if isempty(find(net.add_subunits(j).mult_targets == i,1))
                net.add_subunits(j).mult_targets = ...
                    [net.add_subunits(j).mult_targets, i];
            end
        end
    end
    
    end % method
    
end
    
%% ********************* getting methods **********************************
methods

    function [pred_sigs, gains, subs_out, add_subs_out, mult_subs_comb, ...
              mult_subs_out] = ...
        get_model_internals(net, Xstims, varargin)
    % [pred_sigs, gains, subs_out, add_subs_out, mult_subs_comb, ...
    %  mult_subs_out] = ...
    %       net.get_model_internals(Xstims, varargin)
    %
    % Evaluates current GAM object and returns outputs of additive and
    % multiplicative subunits, as well as overall gain terms (unweighted)
    % acting on the population
    %
    % TODO: It will be useful to update this function to output the 
    % value of the combined gain signals acting on a single additive
    % subunit for each neuron
    %
    % INPUTS:
    %   Xstims:     cell array of T x * matrices
    %
    %   optional key-value pairs:
    %       'indx_tr', vector
    %           subset of 1:T that specifies portion of data used for 
    %           evaluation (default is all data)
    %
    % OUTPUTS:
    %   pred_sigs       T x num_neurons matrix of activity predictions
    %   gains           num_mult_subs x 1 cell array; each cell contains a
    %                   T x num_gain_sigs matrix of the gain signals of a
    %                   given multiplicative subunit, before it is weighted
    %                   by each individual neuron
    %   subs_out        num_add_subs x 1 cell array; each cell contains a
    %                   T x num_neurons matrix of combined add_ and
    %                   mult_subunit outputs before being added (along with
    %                   offset) to produce pred_sigs
    %                   (equivalent to add_subs_out .* mult_subs_comb)
    %   add_subs_out    num_add_subs x 1 cell array; each cell contains a
    %                   T x num_neurons matrix of the output of a given
    %                   additive subunit
    %   mult_subs_comb  num_add_subs x 1 cell array; each cell contains a 
    %                   T x num_neurons matrix of the combined output of 
    %                   all multiplicative subunits acting on the given
    %                   additive subunit
    %   mult_subs_out   num_mult_subs x 1 cell array; each cell contains a 
    %                   T x num_neurons matrix of the output of a given
    %                   multiplicative subunit
    
    % define defaults
    indx_tr = NaN; % NaN means we use all available data

    % parse varargin
    assert(mod(length(varargin), 2) == 0, ...
        'GAM:get_model_internals:Input should be a list of key-value pairs')   
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'indx_tr'
                assert(all(ismember(varargin{i+1}, 1:size(Xstims{1}, 1))), ...
                    'GAM:get_model_internals:Invalid fitting indices')
                indx_tr = varargin{i+1};
            otherwise
                error('GAM:get_model_internals:Invalid input flag "%s"', ...
                    varargin{i});
        end
        i = i + 2;
    end
    
    % use indx_tr and transpose Xstims
    if ~isnan(indx_tr)
        for i = 1:length(Xstims)
            Xstims{i} = Xstims{i}(indx_tr,:);
        end
    end
    T = size(Xstims{1}, 1);
    
    % get internal generating signals for add_subunits
    add_subs_all = cell(length(net.add_subunits),1);
    add_subs_out = cell(length(net.add_subunits),1);
    for i = 1:length(net.add_subunits)
        add_subs_all{i} = net.add_subunits(i).get_model_internals(Xstims);
        % transpose to user orientation
        add_subs_all{i} = cellfun(@(x) x', add_subs_all{i}, ...
                                  'UniformOutput', 0);
        % save final output separately
        add_subs_out{i} = add_subs_all{i}{end};
    end
    
    % get internal generating signals for mult_subunits
    mult_subs_all = cell(length(net.mult_subunits),1);
    mult_subs_out = cell(length(net.mult_subunits),1);
    gains = cell(length(net.mult_subunits),1);
    for i = 1:length(net.mult_subunits)
        mult_subs_all{i} = net.mult_subunits(i).get_model_internals(Xstims);
        % transpose to user orientation
        mult_subs_all{i} = cellfun(@(x) x', mult_subs_all{i}, ...
                                   'UniformOutput', 0);
        % save final output separately
        mult_subs_out{i} = mult_subs_all{i}{end};
        % save unweighted gain signal separately
        if length(mult_subs_all{i}) > 1
            gains{i} = mult_subs_all{i}{end-1};
        end
    end
    
    % multiply output from add_ and mult_subunits
    mult_subs_comb = cell(length(net.add_subunits),1);
    subs_out = cell(length(net.add_subunits),1);
    for i = 1:length(net.add_subunits)
        mult_targets = net.add_subunits(i).mult_targets;
        gain_sig = ones(T, net.num_neurons);
        for j = mult_targets
            gain_sig = gain_sig .* mult_subs_out{j};
        end
        mult_subs_comb{i} = gain_sig;
        subs_out{i} = add_subs_out{i} .* mult_subs_comb{i};
    end
    
    % combine all outputs
    pred_sigs = zeros(T, net.num_neurons);
    for i = 1:length(net.add_subunits)
        pred_sigs = pred_sigs + subs_out{i};
    end
    pred_sigs = bsxfun(@plus, pred_sigs, net.biases');
    pred_sigs = net.apply_spiking_nl(pred_sigs);
    
    end % method
    
    
    function mod_measures = get_gofms(net, true_act, pred_act)
    % mod_measures = net.get_gofms(true_act, pred_act);
    %
    % Evaluates current GAM object using various goodness-of-fit measures
    %
    % INPUTS:
    %   true_act        T x num_neurons matrix of neural activity
    %   pred_act        T x num_neurons matrix of predicted neural activity
    %
    % OUTPUTS:
    %   mod_measures
    %       r2s         num_neurons x 1 vector of r2s
    %       LL          log-likelihood
    %       LLnull      null log-likelihood
    %       cost_func   cost function
    
    T = size(true_act, 1);
    mean_act = ones(T,1) * mean(true_act,1);
    
    switch net.noise_dist
        case 'gauss'
            LL = sum((true_act - pred_act).^2, 1);        
            LLnull = sum((true_act - mean_act).^2, 1);
            LLsat = zeros(1,size(true_act,2));
            Z = 2 * numel(true_act);
        case 'poiss'
            LL = -sum(true_act.*log(pred_act) - pred_act, 1);
            LLnull = -sum(true_act.*log(mean_act) - mean_act, 1);
            LLsat = true_act.*log(true_act);
            LLsat(true_act==0) = 0;
            LLsat = -sum(LLsat - true_act, 1);
            Z = sum(true_act(:));
        otherwise
            error('GAM:get_gofms:Invalid noise distribution "%s"', ...
                net.noise_dist)
    end
    
    mod_measures = struct( ...
        'r2s', (1 - (LLsat-LL)./(LLsat-LLnull))', ...
        'LL', LL', ...
        'LLnull', LLnull', ...
        'LLsat', LLsat', ...
        'cost_func', sum(LL) / Z);

    end % method
   
    
    function reg_pen = get_reg_pen(net)
    % reg_pen = net.get_reg_pen()
    %
    % Retrieves regularization penalties for each subunit of GAM object
    %
    % INPUTS:
    %   none
    %
    % OUTPUTS:
    %   reg_pen: struct containing penalties due to different regs for all
    %            subunits

    reg_pen.add_subunits = struct([]);
    for i = 1:length(net.add_subunits)
        reg_pen.add_subunits = cat(1, reg_pen.add_subunits, ...
                                   net.add_subunits(i).get_reg_pen());
    end
    
    reg_pen.mult_subunits = struct([]);
    for i = 1:length(net.mult_subunits)
        reg_pen.mult_subunits = cat(1, reg_pen.mult_subunits, ...
                                    net.mult_subunits(i).get_reg_pen());
    end
    
    end % method
       
    
    function [mod_meas, mod_ints, mod_reg_pen] = ...
        get_model_eval(net, true_activity, Xstims, varargin)
    % [mod_meas, mod_ints, mod_reg_pen] = ...
    %               net.get_model_eval(true_activity, Xstims, kargs)
    %
    % Evaluates current GAM object and returns relevant model information
    % like goodness-of-fit (pseudo-r2, cost function, etc.), value of 
    % internal model signals, and regularization information
    %
    % INPUTS:
    %   true_activity:  T x num_neurons matrix of neural activity
    %   Xstims:         cell array of T x * matrices
    %
    %   optional key-value pairs:
    %       'indx_tr', vector
    %           subset of 1:T that specifies portion of data used for 
    %           evaluation (default is all data)
    %
    % OUTPUTS:
    %   mod_meas:       struct with goodness-of-fit measures
    %       r2          num_neurons x 1 vector of rsquared vals 
    %       LL          num_neurons x 1 vector of log-likelihood vals
    %       LLnull      num_neurons x 1 vector of null log-likelihood vals
    %       cost_func   scalar value of unregularized cost function
    %   mod_ints:       struct with internal model signals
    %       pred_sigs   T x num_neurons matrix of activity predictions
    %       gains       num_mult_subs x 1 cell array; each cell contains a
    %                   T x num_gain_sigs matrix of the gain signals of a
    %                   given multiplicative subunit, before it is weighted
    %                   by each individual neuron      
    %   mod_reg_pen:    struct containing regularization penalty info     
    %       add_subunits    
    %       mult_subunits
    
    % define defaults
    indx_tr = NaN; % NaN means we use all available data
    assert(mod(length(varargin), 2) == 0, ...
        'GAM:get_model_eval:Input should be a list of key-value pairs')       
    % parse varargin
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'indx_tr'
                assert(all(ismember(varargin{i+1}, 1:size(true_activity, 1))), ...
                    'GAM:get_model_eval:Invalid fitting indices')
                indx_tr = varargin{i+1};
            otherwise
                error('GAM:get_model_eval:Invalid input flag "%s"', ...
                    varargin{i});
        end
        i = i + 2;
    end
   
    % use indx_tr
    if ~isnan(indx_tr)
        true_activity = true_activity(indx_tr,:);
        for i = 1:length(Xstims)
            Xstims{i} = Xstims{i}(indx_tr,:);
        end
    end
    
    % get activation values for all model components
    [mod_ints.pred_sigs, mod_ints.gains] = net.get_model_internals(Xstims);

    % evaluate goodness-of-fit measures
    mod_meas = net.get_gofms(true_activity, mod_ints.pred_sigs);
    
    % get regularization penalites
    mod_reg_pen = net.get_reg_pen();

    end % method
    
end

%% ********************* fitting methods **********************************
methods
    
    function net = fit_model(net, fit_type, true_act, Xstims, varargin)
    % net = net.fit_model(fit_type, true_act, Xstims, kargs)
    %
    % Checks inputs and farms out parameter fitting to other methods 
    % depending on what type of model fit is desired
    %
    % INPUTS:
    %   fit_type:   'add_subunits' | 'mult_subunits' | 'alt'
    %   true_act:   T x num_neurons matrix of neural activity
    %   Xstims:     cell array of T x * input matrices
    %
    %   optional key-value pairs:
    %       'indx_tr', vector
    %           subset of 1:T that specifies portion of data used for 
    %           fitting
    %       'fit_subs', vector
    %           subset of 1:num_add/mult_subunits that specifies which 
    %           subunits to fit; default is all
    %       'fit_layers', cell array of vectors
    %           cell for each fitted subunit that specifies which layers in
    %           that subunit will be fit; default is all. An empty variable
    %           will be expanded to fit all layers in all fitted subunits
    %
    % OUTPUTS:
    %   net:        updated GAM object
    
    % error-check inputs
    assert(ismember(fit_type, {'add_subunits', 'mult_subunits'}), ...
        'GAM:fit_model:Invalid fit_type "%s"', fit_type)
    
    % define defaults
    indx_tr = NaN;      % train on all data
    switch fit_type
        case 'add_subunits'
            num_subunits = length(net.add_subunits);
            fit_subs_add = 1:num_subunits;
            fit_layers = cell(num_subunits, 1);
            for sub = 1:num_subunits
                fit_layers{sub} = ...
                    ones(length(net.add_subunits(sub).layers), 1);
            end
            fit_subs_mult = [];
        case 'mult_subunits'
            num_subunits = length(net.mult_subunits);
            fit_subs_add = [];
            fit_subs_mult = 1:num_subunits;
            fit_layers = cell(num_subunits, 1);
            for sub = 1:num_subunits
                fit_layers{sub} = ...
                    ones(length(net.mult_subunits(sub).layers), 1);
            end
    end
    fit_layers_updated = false;
    
    % parse varargin
    assert(mod(length(varargin), 2) == 0, ...
        'GAM:fit_model:Input should be a list of key-value pairs')    
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'indx_tr'
                assert(all(ismember(varargin{i+1}, 1:size(true_act,1))), ...
                    'GAM:fit_model:Invalid fitting indices')
                indx_tr = varargin{i+1};
            case 'fit_subs'
                assert(all(ismember(varargin{i+1}, 1:num_subunits)), ...
                    'GAM:fit_model:Invalid fit_subs specified')
                switch fit_type
                    case 'add_subunits'
                        fit_subs_add = varargin{i+1}(:)';
                        % redefine fit_layers if necessary
                        if ~fit_layers_updated
                            temp_num_subs = length(fit_subs_add);
                            fit_layers = cell(temp_num_subs, 1);
                            for sub = 1:temp_num_subs
                                curr_sub = fit_subs_add(sub);
                                fit_layers{sub} = ones( ...
                                    length(net.add_subunits(curr_sub).layers), ...
                                    1);
                            end
                        end
                    case 'mult_subunits'
                        fit_subs_mult = varargin{i+1}(:)';
                        % redefine fit_layers if necessary
                        if ~fit_layers_updated
                            temp_num_subs = length(fit_subs_mult);
                            fit_layers = cell(temp_num_subs, 1);
                            for sub = 1:temp_num_subs
                                curr_sub = fit_subs_mult(sub);
                                fit_layers{sub} = ones( ...
                                    length(net.mult_subunits(curr_sub).layers), ...
                                    1);
                            end
                        end
                end                
            case 'fit_layers'
                % TODO: error checking
                if ~isempty(varargin{i+1})
                    fit_layers = varargin{i+1};
                    fit_layers_updated = 1;
                else
                    fit_layers_updated = 0;
                end
            otherwise
                error('GAM:fit_model:Invalid input flag "%s"', varargin{i});
        end
        i = i + 2;
    end
    if ~fit_layers_updated
        switch fit_type
            case 'add_subunits'
                temp_num_subs = length(fit_subs_add);
                fit_layers = cell(temp_num_subs, 1);
                for sub = 1:temp_num_subs
                    curr_sub = fit_subs_add(sub);
                    fit_layers{sub} = ones( ...
                        length(net.add_subunits(curr_sub).layers), ...
                        1);
                end
            case 'mult_subunits'
                temp_num_subs = length(fit_subs_mult);
                fit_layers = cell(temp_num_subs, 1);
                for sub = 1:temp_num_subs
                    curr_sub = fit_subs_mult(sub);
                    fit_layers{sub} = ones( ...
                        length(net.mult_subunits(curr_sub).layers), ...
                        1);
                end
        end  
    end
    
    % use indx_tr and transpose input
    if ~isnan(indx_tr)
        for i = 1:length(Xstims)
            Xstims{i} = Xstims{i}(indx_tr,:)';
        end
        true_act = true_act(indx_tr,:)';
    else
        for i = 1:length(Xstims)
            Xstims{i} = Xstims{i}';
        end
        true_act = true_act';
    end
    
    % create fitting struct to pass to other fitting methods
    fit_struct = struct( ...
        'fit_type', fit_type, ...
        'true_act', true_act, ...
        'fit_subs_add', fit_subs_add, ...
        'fit_subs_mult', fit_subs_mult, ...
        'fit_layers', {fit_layers});
    fit_struct.Xstims = Xstims; % turns fit_struct into a struct array if added above...
    clear true_act Xstims  % free up memory

    % check consistency between inputs
    net.check_fit_struct(fit_struct); 
    
    % pretrain model
    if strcmp(fit_type, 'add_subunits')
        for i = 1:length(fit_subs_add)
            curr_sub = fit_subs_add(i);
            if ~strcmp(net.add_subunits(curr_sub).pretraining, 'none')
                net.add_subunits(curr_sub) = ...
                    net.add_subunits(curr_sub).pretrain( ...
                        fit_struct.Xstims, fit_layers{i});
            end
        end
    elseif strcmp(fit_type, 'mult_subunits')
        for i = 1:length(fit_subs_mult)
            curr_sub = fit_subs_mult(i);
            if ~strcmp(net.mult_subunits(curr_sub).pretraining, 'none')
                net.mult_subunits(curr_sub) = ...
                    net.mult_subunits(curr_sub).pretrain( ...
                        fit_struct.Xstims, fit_layers{i});
            end
        end
    end
    
    % fit specified params
    if strcmp(fit_type, 'add_subunits')
        % update fit_subs_mult for fit_history
        fit_struct.fit_subs_mult = [];
        net = net.fit_add_subunits(fit_struct);
    elseif strcmp(fit_type, 'mult_subunits')
        % update fit_subs_add for fit_history
        fit_struct.fit_subs_add = [];
        % format network so that fit_add_subunits can be used
        [net0, fit_struct0] = net.format_for_mult_fit(fit_struct);
        % fit model
        net0 = net0.fit_add_subunits(fit_struct0);
        % reformat network
        net = net.format_from_mult_fit(net0, fit_struct);
    end
    
    end % method
        
end

%% ********************* hidden methods ***********************************
methods (Hidden)
  
    function check_fit_struct(net, fit_struct)
    % net.check_fit_struct(fit_struct)
    %
    % Checks input structure for fitting methods
    %
    % INPUTS:
    %   fit_struct: structure for parameter fitting; see fit_model method
    %               for relevant fields
    %
    % OUTPUTS:
    %   none; throws flag if error
    %
    % CALLED BY:
    %   GAM.fit_model
    
    % check that input_target, input_params and Xstims are compatible for
    % each additive subunit
    for i = 1:length(net.add_subunits)
        targ = net.add_subunits(i).input_target;
        dims = net.add_subunits(i).input_params.dims;
        assert(size(fit_struct.Xstims{targ},1) == prod(dims), ...
            ['GAM:fit_model:', ...
             'Xstims and input params inconsistent for add_subunit %g'], i)
    end
    
    % check that input_target, input_params and Xstims are compatible for
    % each multiplicative subunit
    for i = 1:length(net.mult_subunits)
        targ = net.mult_subunits(i).input_target;
        dims = net.mult_subunits(i).input_params.dims;
        assert(size(fit_struct.Xstims{targ},1) == prod(dims), ...
            ['GAM:fit_model:', ...
             'Xstims and input params inconsistent for mult_subunit %g'], i)
    end
    
    % check that Xstims have same time dimension
    assert(length(unique(cellfun(@(x) size(x,2), fit_struct.Xstims))) == 1, ...
        'GAM:fit_model:Time dim not consisent across Xstims')
    
    % check that Xstims and true_act have same time dimension
    assert(size(fit_struct.true_act,2) == size(fit_struct.Xstims{1},2), ...
        'GAM:fit_model:Time dim not consisent across true_act and Xstims')
   
    end % method
  
    
    function net = update_fit_history(net, fit_struct, params, f_val, output)
    % net = net.update_fit_history(fit_struct, params, func_val, output)
    %
    % INPUTS:
    %   fit_struct: structure for parameter fitting; see fit_model method
    %               for relevant fields
    %   params:     params returned by opt routine
    %   f_val:   final function value returned by opt routine
    %   output:     output structure returned by opt routine
    %
    % OUTPUTS:
    %   net:        updated GAM object

    curr_fit_details = struct( ...
        'fit_subs_add', fit_struct.fit_subs_add, ...
        'fit_subs_mult', fit_struct.fit_subs_mult, ...
        'func_val', f_val, ...
        'func_vals', output.trace.fval, ...
        'iters', output.iterations, ...
        'first_order_opt', output.firstorderopt, ...
        'exit_msg', output.message, ...
        'params_fit', length(params));
    
    net.fit_history = cat(2, net.fit_history, curr_fit_details);

    end % method
    
    
    function sig = apply_spiking_nl(net, sig)
    % sig = net.apply_spiking_nl(sig)
    %
    % Applies spiking nonlinearity to given input
    %
    % INPUTS:
    %   sig:    matrix of input values
    %
    % OUTPUTS:
    %   sig:    input passed through spiking nonlinearity

    switch net.spiking_nl
        case 'lin'
        case 'relu'
            sig = max(0, sig);
        case 'sigmoid'
            sig = 1 ./ (1 + exp(-sig));
        case 'softplus'
            temp_sig = log(1 + exp(sig));
            % take care of overflow - appx linear
            temp_sig(sig > net.max_g) = sig(sig > net.max_g);
            % take care of underflow so LL is defined (taking logs later)
            temp_sig(temp_sig < net.min_pred_rate) = net.min_pred_rate;
            sig = temp_sig;
        case 'exp'
            sig = exp(sig);
    end
    
    end % method

    
    function sig = apply_spiking_nl_deriv(net, sig)
    % sig = net.apply_spiking_nl_deriv(sig)
    %
    % Calculates the derivative of the spiking nonlinearity to given input
    %
    % INPUTS:
    %   sig:      matrix of input values
    %
    % OUTPUTS:
    %   sig:      input passed through derivative of spiking nonlinearity

    switch net.spiking_nl
        case 'lin'
            sig = ones(size(sig));
        case 'relu'
            if 1
                sig = relu_deriv_inplace(sig);
            else
                sig(sig <= 0) = 0; sig(sig > 0) = 1;
            end
        case 'sigmoid'
            sig = exp(-sig) ./ (1 + exp(-sig)).^2;
        case 'softplus'
            % temp_sig = exp(sig) ./ (1 + exp(sig));
            % % e^x/(1+e^x) => 1 for large x
            % temp_sig(sig > net.max_g) = 1; 
            % sig = temp_sig;
            sig = 1 ./ (1 + exp(-sig)); % ~twice as fast          
        case 'oneplus'
            sig = ones(size(sig));
        case 'exp'
            sig = exp(sig);
    end
    
    end % method
    
    
    function [net0, fit_struct0] = format_for_mult_fit(net, fit_struct)
    % net.format_for_mult_fit(fit_struct)
    %
    % Takes an GAM object and reformats it into a new GAM object so that
    % fit_add_subunits can be called for fitting mult_subunits. Allows
    % reuse of fitting code that would otherwise have to be slightly 
    % different for the two fitting scenarios.
    %
    % INPUTS:
    %   fit_struct: structure for parameter fitting; see fit_model method
    %               for relevant fields
    %
    % OUTPUTS:
    %   net0:       new GAM object
    
    fit_struct0 = fit_struct;
    
    % pull out useful constants
    fit_subs = fit_struct.fit_subs_mult;
    num_neurons = net.num_neurons;
    num_stims = length(fit_struct.Xstims);
    T = size(fit_struct.Xstims{1},2);
    
    % error-check inputs; make sure that two mult_subunits acting on same
    % add_subunit will not be fit together
    add_targs = [];
    for i = fit_subs
        add_targs = [add_targs; net.mult_subunits(i).add_targets(:)];
        assert(length(unique(add_targs)) == length(add_targs), ...
            ['GAM:format_for_mult_fit:', ...
            'Two mult_subunits cannot target same add_subunit'])
    end
    
    % calculate model internals
    [~, ~, subs_out, add_subs_out, ~, mult_subs_out] = ...
            net.get_model_internals(cellfun(@(x) x', ...
                                    fit_struct.Xstims, ...
                                    'UniformOutput', 0));
    
    % create new GAM
    net0 = GAM('noise_dist', net.noise_dist, 'spiking_nl', net.spiking_nl);
    
    % copy over additional data
    net0.num_neurons = net.num_neurons;
    net0.biases = net.biases;
    net0.optim_params = net.optim_params;
    
    % create new subunits
    count = 1;
    for i = fit_subs
        
        % -----------------------------------------------------------------
        % turn fit_sub into add_subunit
        % -----------------------------------------------------------------
        % create dummy subunit
        net0 = net0.create_subunit('add', num_neurons, [], ...
                    1, net.mult_subunits(i).input_params);
        % replace with mult_subunit
        net0.add_subunits(count) = net.mult_subunits(i);
        net0.add_subunits(count).add_targets = [];
        net0.add_subunits(count).mult_targets = count;
        
        % -----------------------------------------------------------------
        % turn add_targets of fit sub, along with their other mult_subunits
        % into a single signal
        % -----------------------------------------------------------------
        mult_sig = zeros(num_neurons, T);
        % loop over add_targets for this mult_subunit
        for j = net.mult_subunits(i).add_targets
            temp_sig = add_subs_out{j}';
            % include output from mult_subunits that are not being fit
            for k = net.add_subunits(j).mult_targets
                if k ~= i
                    % different mult subunit from one being fit
                    temp_sig = temp_sig .* mult_subs_out{k}';
                end
            end
            mult_sig = mult_sig + temp_sig;
        end
        
        % put signal into Xstims
        input_params = GAM.create_input_params([1, num_neurons, 1]);
        fit_struct0.Xstims{end+1} = GAM.create_time_embedding( ...
                            mult_sig', input_params)';
        num_stims = num_stims + 1;
        
        % -----------------------------------------------------------------
        % create a mult_subunit for this new signal
        % -----------------------------------------------------------------
        % create dummy subunit
        net0 = net0.create_subunit('mult', num_neurons, count, ...
                num_stims, input_params, ...
                'act_funcs', 'lin');
        % don't transform Xstim
        net0.mult_subunits(count).layers.weights = eye(num_neurons);
        net0.mult_subunits(count).layers.biases = zeros(num_neurons,1);
               
        count = count + 1;
        
    end
     
    % ---------------------------------------------------------------------
    % turn all non-targeted add_subunits into single signal/add_subunit
    % ---------------------------------------------------------------------
    nontarg_add_subs = setdiff(1:length(net.add_subunits), add_targs);
    
    if ~isempty(nontarg_add_subs)
        % get signal
        temp_sig = zeros(num_neurons, T);
        for i = nontarg_add_subs
            temp_sig = temp_sig + subs_out{i}';
        end
        
        % put signal into Xstims
        input_params = GAM.create_input_params([1, num_neurons, 1]);
        fit_struct0.Xstims{end+1} = GAM.create_time_embedding( ...
                            temp_sig', input_params)';
        num_stims = num_stims + 1;
        
        % create an add_subunit for this new signal
        % create dummy subunit
        net0 = net0.create_subunit('add', num_neurons, [], ...
                num_stims, input_params, ...
                'act_funcs', 'lin');
        % don't transform Xstim
        net0.add_subunits(count).layers.weights = eye(num_neurons);
        net0.add_subunits(count).layers.biases = zeros(num_neurons,1);
    end
    
    % update fit_struct0
    fit_struct0.fit_type = 'add';
	fit_struct0.fit_subs_add = 1:length(fit_subs);
    fit_struct0.fit_subs_mult = [];
    
    end % method
    
    
    function net = format_from_mult_fit(net, net_mult, fit_struct)
    % net.format_from_mult_fit(net_mult, fit_struct)
    %
    % Takes a formatted GAM object after fitting add_subnits and updates
    % the original GAM object.
    %
    % INPUTS:
    %   net_mult:   GAM object resulting from GAM.format_for_mult_fit
    %   fit_struct: structure for parameter fitting; see fit_model method
    %               for relevant fields
    %
    % OUTPUTS:
    %   net:       updated GAM object
    
    fit_subs = fit_struct.fit_subs_mult;
    
    % transfer params from add_subunits back to mult_subunits
    for i = 1:length(fit_subs)
        net.mult_subunits(fit_subs(i)).layers = ...
            net_mult.add_subunits(i).layers;
    end
    
    % update remaining network properties
    net.biases = net_mult.biases;
    net.fit_history = cat(2, net.fit_history, net_mult.fit_history);
    net.fit_history(end).fit_subs_add = [];
    net.fit_history(end).fit_subs_mult = fit_subs;
    
    end % method
    
end

%% ********************* static methods ***********************************
methods (Static)
    
	function input_params = create_input_params(dims, varargin)
    % input_params = GAM.create_input_params(dims, kargs)
    %
    % Creates a struct containing stimulus parameters
    %
    % INPUTS:
    %   dims:           dimensionality of the (time-embedded) stimulus, in 
    %                   the form: [num_lags num_xpix num_ypix]. For 1 
    %                   spatial dimension use only num_xpix
    %     
    %   optional key-value pairs:
    %       'tent_spacing', scalar
    %           spacing of tent-basis functions when using a tent-basis 
    %           representaiton of the stimulus (allows for the stimulus 
    %           filters to be represented at a lower time resolution than 
    %           other model components). 
    %       'boundary_conds', vector
    %           boundary conditions on each dimension (Inf is free, 0 is 
    %           tied to 0, and -1 is periodic)
    %
    % OUTPUTS:
    %   input_params: struct of stimulus parameters
    
    % Set defaults
    tent_spacing = [];          % default no tent-bases
    boundary_conds = [0 0 0];   % tied to 0 in all dims

    % parse varargin
    assert(mod(length(varargin), 2) == 0, ...
        'GAM:creat_input_params:Input should be a list of key-value pairs')
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'tent_spacing'
                tent_spacing = varargin{i+1};
            case 'boundary_conds'
                for j = 1:length(varargin{i+1})
                    assert(ismember(varargin{i+1}(j), [Inf, 0, -1]), ...
                        ['GAM:creat_input_params:', ...
                         'Invalid boundary condition specified'])
                end
                boundary_conds = varargin{i+1};
            otherwise
                error('GAM:creat_input_params:Invalid input flag "%s"', ...
                    varargin{i}); 
        end	
        i = i + 2;
    end

    % Make sure stim_dims input has form [num_lags num_xpix num_ypix] and
    % concatenate with 1's if necessary    
    while length(dims) < 3 
        % pad dims with 1s for book-keeping
        dims = cat(2, dims, 1);
    end

    % update matching boundary conditions
    while length(boundary_conds) < 3
        % assume free boundaries on spatial dims if not specified
        boundary_conds = cat(2, boundary_conds, 0); 
    end

    % create struct to output
    input_params = struct( ...
        'dims', dims, ...
        'tent_spacing', tent_spacing, ...
        'boundary_conds',boundary_conds);

    end % method
	
    
    function Xmat = create_time_embedding(input, input_params)
    % Xmat = GAM.create_time_embedding(input, input_params)
    %
    % Takes a Txd stimulus matrix and creates a time-embedded matrix of
    % size Tx(d*num_lags), where num_lags is the desired number of time
    % lags specified in the stim_params struct. If stim is a 3d array the
    % spatial dimensions are folded into the 2nd dimension. Assumes
    % zero-padding. Note that Xmat is formatted so that adjacent time lags
    % are adjacent within a time-slice of Xmat. Thus Xmat(t,1:num_lags)
    % gives all the time lags of the first spatial pixel at time t.
    %
    % INPUTS:
    %   input:   T x * input matrix
    %   input_params:
    %
    % OUTPUTS:
    %   Xmat:   time-embedded input matrix
    
    sz = size(input);

    % if there are two spatial dims, fold them into one
    if length(sz) > 2
        input = reshape(input, sz(1), prod(sz(2:end)));
    end
    
    % no support for more than two spatial dims
    if length(sz) > 3
        warning(['GAM:create_tim_embedding:', ...
                 'More than two spatial dimensions not supported; ', ...
                 'creating Xmat anyways...']);
    end

    % check that the size of input matches with the specified input_params
    % structure
    [T, num_pix] = size(input);
    if prod(input_params.dims(2:end)) ~= num_pix
        error('GAM:create_time_embedding:Stimulus dimension mismatch');
    end
    
    % if using a tent-basis representation
    if ~isempty(input_params.tent_spacing)
        tbspace = input_params.tent_spacing;
        % create a tent-basis (triangle) filter
        tent_filter = [(1:tbspace) / tbspace ...
                        1 - (1:tbspace - 1) / tbspace] / tbspace;

        % apply to the stimulus
        filtered_stim = zeros(size(input));
        for i = 1:length(tent_filter)
            filtered_stim = filtered_stim + ...
                GAM.shift_mat_zpad(input, i-tbspace, 1) * tent_filter(i);
        end

        input = filtered_stim; 
        lag_spacing = tbspace;
    else
        lag_spacing = 1;
    end

    % for temporal only stimuli (this method can be faster if you're not 
    % using tent-basis rep
    if num_pix == 1
        Xmat = toeplitz(input, ...
            [input(1) zeros(1, input_params.dims(1) - 1)]);
    else
        % otherwise loop over lags and manually shift the stim matrix
        Xmat = zeros(T, prod(input_params.dims));
        for n = 1:input_params.dims(1)
            Xmat(:,n-1+(1:input_params.dims(1):(num_pix*input_params.dims(1)))) = ...
                GAM.shift_mat_zpad(input, lag_spacing * (n-1), 1);
        end
    end
    
    end % method
        
end

%% ********************* static hidden methods ****************************
methods (Static, Hidden)
    
    function Xshifted = shift_mat_zpad(X, shift, dim)
    % Xshifted = shift_mat_zpad(X, shift, <dim>)
    %
    % Takes a vector or matrix and shifts it along dimension dim by amount
    % shift using zero-padding. Positive shifts move the matrix right or 
    % down
    %
    % INPUTS:
    %   X:          matrix or vector to shift
    %   shift:      amount to shift by. positive shifts move the matrix right
    %               or down
    %   <dim>:      optional; dimension to shift along
    %
    % OUTPUTS:
    %   Xshifted:   shifted matrix or vector

    % default to appropriate dimension if X is one-dimensional
    if nargin < 3
        [a,~] = size(X);
        if a == 1
            dim = 2;
        else
            dim = 1;
        end
    end

    sz = size(X);
    if dim == 1
        if shift >= 0
            Xshifted = [zeros(shift,sz(2)); X(1:end-shift,:)];
        else
            Xshifted = [X(-shift+1:end,:); zeros(-shift,sz(2))];
        end
    elseif dim == 2
        if shift >= 0
            Xshifted = [zeros(sz(1),shift) X(:,1:end-shift)];
        else
            Xshifted = [X(:,-shift+1:end) zeros(sz(1),-shift)];
        end
    end
    
    end % method

    
    function optim_params = set_init_optim_params()
    % optim_params = GAM.set_init_optim_params();
    %
    % Sets default optimization parameters for the various optimization 
    % routines

    % optim_params
    optim_params.optimizer   = 'minFunc'; % opt package to use
    optim_params.Display     = 'off';     % opt routine output
    optim_params.monitor     = 'iter';    % save opt routine output
    optim_params.deriv_check = 0;

    % both matlab and mark schmidt options
    optim_params.max_iter    = 1000;        
    optim_params.maxIter     = 1000;        
    optim_params.maxFunEvals = 2000;      % mostly for sd in minFunc
    optim_params.optTol      = 1e-6;      % tol on first order optimality (max(abs(grad))
    optim_params.progTol     = 1e-16;     % tol on function/parameter values
    optim_params.TolX        = 1e-10;     % tol on progress in terms of function/parameter changes
    optim_params.TolFun      = 1e-7;      % tol on first order optimality

    % just mark schmidt options
    optim_params.Method = 'lbfgs';

    % just matlab options
    optim_params.Algorithm  = 'quasi-newton';
    optim_params.HessUpdate = 'steepdesc'; % bfgs default incredibly slow
    optim_params.GradObj    = 'on';
    optim_params.DerivativeCheck = 'off';
    optim_params.numDiff    = 0;

    end % method
    
end
   
end



classdef GAMLayer
    
% Class implementing an individual layer of a Subunit object for the 
% generalized affine model (GAM)
%
% Author: Matt Whiteway
%   08/31/17

properties
    weights                 % matrix of weights
    biases                  % vector of biases
    act_func                % string specifying layer activation function
    reg_lambdas             % struct defining regularization hyperparameters
    init_params             % struct saving initialization parameters
      % rng_state
      % init_weights
end

properties (Hidden)
    min_pred_rate = 1e-5;   % min val for logs
    max_g = 50;             % max val for exponentials
    
    % user options
    allowed_act_funcs = ...
        {'lin', ...         % f(x) = x
         'relu', ...        % f(x) = max(0, x)
         'sigmoid', ...     % f(x) = 1 / (1 + exp(-x))
         'softplus', ...    % f(x) = log(1 + exp(x))
         'oneplus', ...     % f(x) = 1 + x
         'exp' ...          % f(x) = exp(x)
        };
    allowed_regtypes = ...
        {'l2_weights', 'l2_biases', ...
         'l1_weights', 'l1_biases'};
    allowed_init_types = ...
        {'gauss', ...
         'trunc_gauss', ...
         'uniform', ...
         'orth', ...
         'zeros'};    
end

%% ********************  constructor **************************************
methods
      
    function layer = GAMLayer(num_out, num_in, init_method, varargin)
    % layer = GAMLayer(num_out, num_in, init_method, kargs);
    % 
    % Constructor for GAMLayer class
    %
    % INPUTS:
    %   num_in:         number of input nodes
    %   num_out:        number of output nodes
    %   init_method:    string specifying a random initialization; see
    %                   allowed_init_types for supported options
    %                   or a 2x1 cell array containt weights and biases of
    %                   appropriate dimensions
    %
    %   optional key-value pairs
    %       'act_func', string
    %           specifies activation function for layer; see 
    %           allowed_act_funcs for supported options (default: relu)
    %
    % OUTPUTS:
    %   layer: initialized GAMLayer object
    
    if nargin == 0
        % handle the no-input-argument case by returning a null model. This
        % is important when initializing arrays of objects
        return
    end

    % error-check inputs
    assert(num_out > 0, ...
        'GAM:GAMLayer:Constructor:Layer must have positive number of inputs')
    assert(num_out > 0, ...
        'GAM:GAMLayer:Constructor:Layer must have positive number of outputs')
    assert(ismember(init_method, layer.allowed_init_types),...
        'GAM:GAMLayer:Constructor:Invalid init_method "%s"', init_method)
    
    % define defaults
    act_func_ = 'relu';
    
    % parse varargin
    assert(mod(length(varargin), 2) == 0, ...
        'GAM:GAMLayer:Constructor:Input should be a list of key-value pairs')
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'act_func'
                assert(ismember(varargin{i+1}, layer.allowed_act_funcs),...
                    ['GAM:GAMLayer:Constructor:', ...
                     'Invalid activation function "%s"'], varargin{i+1})
                act_func_ = varargin{i+1};
            otherwise
                error('GAM:GAMLayer:Constructor:Invalid input flag "%s"', ...
                    varargin{i});
        end
        i = i + 2;
    end
    
    % initialize weights
    [weights_, biases_, init_params_] = GAMLayer.set_init_weights_stat(...
        init_method, num_in, num_out);
        
    % set properties
    layer.weights = weights_;
    layer.biases = biases_;
    layer.act_func = act_func_;
    layer.reg_lambdas = GAMLayer.init_reg_lambdas(); % init all to 0s
    layer.init_params = init_params_;
    
    end

end

%% ********************  setting methods **********************************
methods
    
    function layer = set_reg_params(layer, varargin)
    % layer = layer.set_reg_params(kargs)
    %
    % Takes a sequence of key-value pairs to set regularization parameters 
    % for a GAMLayer object
    %
    % INPUTS:
    %   optional key-value pairs:
    %       'reg_type', scalar
    %           'l2_weights' | 'l2_biases' | 'l1_weights' | 'l1_biases'
    %
    % OUTPUTS:
    %   layer: updated GAMLayer object
    
    % parse varargin
    assert(mod(length(varargin), 2) == 0, ...
        'GAM:GAMLayer:set_reg_params:Input should be a list of key-value pairs')
    i = 1;
    while i <= length(varargin)
        assert(varargin{i+1} >= 0, ...
            'GAM:GAMLayer:set_reg_params:reg value must be nonnegative')
        switch lower(varargin{i})
			case 'l2_weights'    
                layer.reg_lambdas.l2_weights = varargin{i+1};
            case 'l2_biases'
                layer.reg_lambdas.l2_biases = varargin{i+1};
            case 'l1_weights'
                layer.reg_lambdas.l1_weights = varargin{i+1};
            case 'l1_biases'
                layer.reg_lambdas.l1_biases = varargin{i+1};
            otherwise
                error('GAM:GAMLayer:set_reg_params:Invalid input flag "%s"', ...
                    varargin{i});
        end
        i = i + 2;
    end
    
    end % method
    
    
    function layer = set_init_weights(layer, init_weights)
    % layer = layer.set_init_weights(init_weights)
    %
    % Sets weights and biases properties of GAMLayer object
    %
    % INPUTS:
    %   init_weights:   string specifying a random initialization; see
    %                   allowed_init_types for supported options
    %                   or a 2x1 cell array containt weights and biases of
    %                   appropriate dimensions
    %
    % OUTPUT:
    %   layer:          updated GAMLayer object
    
    % call static set_init_weights used in constructor
    [weights_, biases_, init_params_] = GAMLayer.set_init_weights_stat(...
                                          init_weights, ...
                                          layer.num_in, ...
                                          layer.num_out);

    % set properties
    layer.weights = weights_;
    layer.biases = biases_;
    layer.init_params = init_params_;
    
    end % method
    
end

%% ********************  getting methods **********************************
methods
    
    function reg_pen = get_reg_pen(layer)
    % reg_pen = layer.get_reg_pen()
    %
    % Retrieves regularization penalties on layer weights and biases
    %
    % INPUTS:
    %   none
    %
    % OUTPUTS:
    %   reg_pen: struct containing penalties due to different regs
    
    % set aside constants to keep things clean
    lambda_l2_w = layer.reg_lambdas.l2_weights;
    lambda_l2_b = layer.reg_lambdas.l2_biases;
    lambda_l1_w = layer.reg_lambdas.l1_weights;
    lambda_l1_b = layer.reg_lambdas.l1_biases;
    
    % get penalty terms
    % L2 on weights
    reg_pen.l2_weights = 0.5 * lambda_l2_w * sum(layer.weights(:).^2); 
    % L2 on biases
    reg_pen.l2_biases = 0.5 * lambda_l2_b * sum(layer.biases.^2);
    % L1 on weights
    reg_pen.l1_weights = lambda_l1_w * sum(abs(layer.weights(:))); 
    % L1 on biases
    reg_pen.l2_biases = lambda_l1_b * sum(abs(layer.biases));
    
    end % method
    
end

%% ********************  hidden methods ***********************************
methods (Hidden)
    
    function sig = apply_act_func(layer, sig)
    % sig = layer.apply_act_func(sig)
    %
    % Internal function that applies activation function of the nodes
    % in the layer to a given input
    %
    % INPUTS:
    %   sig:    num_in x T matrix
    %
    % OUTPUTS:
    %   sig:    input passed through activation function

    switch layer.act_func
        case 'lin'
        case 'relu'
            sig = max(0, sig);
        case 'sigmoid'
            sig = 1 ./ (1 + exp(-sig));
        case 'softplus'
            temp_sig = log(1 + exp(sig));
            % take care of overflow - appx linear
            temp_sig(sig > layer.max_g) = sig(sig > layer.max_g);
            % take care of underflow so LL is defined (taking logs later)
            temp_sig(temp_sig < layer.min_pred_rate) = layer.min_pred_rate;
            sig = temp_sig;
        case 'oneplus'
            sig = 1 + sig;
        case 'exp'
            sig = exp(sig);
    end
    
    end % method

    
    function sig = apply_act_deriv(layer, sig)
    % sig = layer.apply_act_deriv(sig)
    %
    % internal function that calculates the derivative of the activation 
    % function of the nodes in the layer to a given input
    %
    % INPUTS:
    %   sig:      num_in x T matrix
    %
    % OUTPUTS:
    %   sig:      input passed through derivative of activation function

    switch layer.act_func
        case 'lin'
            sig = ones(size(sig));
        case 'relu'
            if 1
                sig = relu_deriv_inplace(sig);
            else
                sig(sig <= 0) = 0; sig(sig > 0) = 1;
            end
        case 'sigmoid'
            temp_sig = exp(-sig) ./ (1 + exp(-sig)).^2;
            % e^(-x)/(1+e^(-x))^2 => 0 for large abs(x)
            temp_sig(abs(sig) > layer.max_g) = 0;
            sig = temp_sig;
        case 'softplus'
            % temp_sig = exp(sig) ./ (1 + exp(sig));
            % % e^x/(1+e^x) => 1 for large x
            % temp_sig(sig > layer.max_g) = 1; 
            % sig = temp_sig;
            sig = 1 ./ (1 + exp(-sig)); % ~twice as fast
        case 'oneplus'
            sig = ones(size(sig));
        case 'exp'
            sig = exp(sig);
    end
    
    end % method
    
end

%% ********************  static methods ***********************************
methods (Static)
    
    function reg_lambdas = init_reg_lambdas()
    % reg_lambdas = GAMLayer.init_reg_lambdas()
    %
    % creates reg_lambdas struct and sets default values to 0; called from
    % GAMLayer constructor
    %
    % INPUTS:
    %   none
    %
    % OUTPUTS:
    %   reg_lambdas: struct containing initialized reg params

    reg_lambdas.l2_weights = 0;     % L2 on weights
    reg_lambdas.l2_biases = 0;      % L2 on biases
    reg_lambdas.l1_weights = 0;     % L1 on weights
    reg_lambdas.l1_biases = 0;      % L1 on biases
    
    end % method
 
    
    function [weights, biases, init_params] = set_init_weights_stat( ...
        init_method, num_in, num_out)
                                            
    % [weights, biases, init_params] = GAMLayer.set_init_weights_stat( ...
    %   init_method, num_in, num_out)
    % 
    % static function that initializes weights/biases and sets init_params
    % structure based on input. Called from the GAMLayer constructor and from
    % the non-static method set_init_weights
    %
    % INPUTS:
    %   init_method:    string specifying a random initialization; see
    %                   allowed_init_types for supported options
    %                   or a 2x1 cell array containt weights and biases of
    %                   appropriate dimensions
    %   num_in:         number of input nodes
    %   num_out:        number of output nodes
    %
    % OUTPUTS:
    %   weights:        num_out x num_in weight matrix
    %   biases:         num_out x 1 bias vector
    %   init_params:    struct specifying init_weights and rng_state
    
    if ischar(init_method)
        
        init_params.rng_state = rng();
        init_params.init_weights = lower(init_method);
        
        % randomly initialize weights; start biases off at 0
        s = 0.1; %1/sqrt(num_in);
        switch lower(init_method)
            case 'gauss'
                weights = s * randn(num_out, num_in);
            case 'trunc_gauss'
                weights = abs(s * randn(num_out, num_in));
            case 'uniform'
                % choose weights uniformly from the interval [-r, r]
                r = 4 * sqrt(6) / sqrt(num_out + num_in + 1);   
                weights = rand(num_out, num_in) * 2 * r - r;
            case 'orth'
                temp = s * randn(num_out, num_in);
                if num_in >= num_out
                    [u, ~, ~] = svd(temp');
                    weights = u(:, 1:num_out)';
                else
                    weights = temp;
                end
            case 'zeros'
                weights = zeros(num_out, num_in);
            otherwise
                error(['GAM:GAMLayer:set_init_weights_stat:', ...
                       'Invalid init string "%s"'], init_method)
        end
        biases = zeros(num_out, 1);

    elseif iscell(init_method)
        
        % use 'init_weights' to initialize weights
        assert(size(init_method{1}) == [num_out, num_in], ...
            ['GAM:GAMLayer:set_init_weights_stat:', ...
             'Weight matrix has improper dimensions']);
        assert(size(init_method{2}) == [num_out, 1], ...
            ['GAM:GAMLayer:set_init_weights_stat:', ...
             'Weight matrix has improper dimensions']);

        % get parameters
        weights = init_method{1};
        biases = init_method{2};
        
        init_params.rng_state = NaN;
        init_params.init_weights = 'user-supplied';
        
    else
        error(['GAM:GAMLayer:set_init_weights_stat:', ...
               'init_weights must be a string or a cell array'])
    end
    
    end % method

end

end
classdef GAMSubunit
    
% Class implementing the subunits of an GAM object
%
% Author: Matt Whiteway
%   08/31/17

properties
    add_targets     % add_subunit: empty
                    % mult_subunit: list of targeted additive subunits
    mult_targets    % add_subunit: list of multiplicative subunits that use
                    % this subunit as a target
                    % mult_subunit: empty
    input_target    % specifies which input matrix the subunit acts on
    input_params    % struct defining input params for input_target
        % dims
        % tent_spacing
        % boundary_conds
    layers          % array of layer objects comprising subunit
    pretraining     % string specifying type of pretraining for subunit
end

properties (Hidden) 
    % user options
    allowed_pretrainers = {'none', 'pca', 'pca-varimax'};
end

%% ********************  constructor **************************************
methods
      
    function subunit = GAMSubunit(layer_sizes, add_targets, mult_targets, ...
                               input_target, input_params, varargin)
    % subunit = GAMSubunit(layer_sizes, add_targets, mult_targets, ...
    %                   input_target, input_params, kargs)
    %
    % Constructor function for a GAMSubunit object
    %
    % INPUTS:
    %   layer_sizes:    scalar array of number of nodes in each layer,
    %                   excluding the input layer
    %   add_targets:    if add_subunit: empty
    %                   if mult_subunit: list of additive targets
    %   mult_targets:   if add_subunit: list of targeting mult subunits
    %                   if mult_subunit: empty
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
    %       'pretraining', string
    %           ['none'] | 'pca' | 'pca-varimax' |
    %           specifies how subunit weights are modified before training
    %
    % OUTPUT:
    %   subunit: updated GAMSubunit object
    
    if nargin == 0
        % handle the no-input-argument case by returning a null model. This
        % is important when initializing arrays of objects
        return 
    end

    % error-check inputs
    assert(~isempty(layer_sizes), ...
        'GAM:GAMSubunit:Constructor:Must specify at least one layer')
    assert(mod(length(varargin), 2) == 0, ...
        'GAM:GAMSubunit:Constructor:Input should be a list of key-value pairs')

    % define defaults    
    act_funcs = repmat({'relu'}, 1, length(layer_sizes));
    act_funcs{end} = 'lin'; % for 'gauss' noise dist default in GAM
    init_weights = repmat({'gauss'}, 1, length(layer_sizes));
    pretraining = 'none';

    % parse varargin
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'act_funcs'
                % error-checking for valid type occurs in Layer constructor
                act_funcs = varargin{i+1};
                % if act_funcs is specified as a single string, change to 
                % cell array with single entry
                if ~iscell(act_funcs) && ischar(act_funcs)
                    act_funcs = cellstr(act_funcs);
                end
                % handle different input formats
                if length(act_funcs) == 1 && length(layer_sizes) > 1
                    % single entry - default to using this act_func for all
                    % layers
                    act_funcs = repmat(act_funcs,1,length(layer_sizes));
                elseif length(act_funcs) ~= length(layer_sizes)
                    % multiple entries - verify that user supplied correct
                    % number of entries
                    error(['GAM:GAMSubunit:Constructor:', ...
                           'Invalid number of act_funcs'])
                end
            case 'init_params'
                % handle different input formats (error-checking for valid
                % type occurs in Layer constructor)
                if ischar(varargin{i+1})
                    % input is a single string
                    init_weights = varargin{i+1};
                    init_weights = cellstr(init_weights);
                    % default to using this string for all layers
                    if length(layer_sizes) > 1
                        init_weights = repmat(init_weights,1,length(layer_sizes));
                    end
                elseif iscell(varargin{i+1}) && ischar(varargin{i+1}{1})
                    % input is a cell array of strings
                    init_weights = varargin{i+1};
                    if length(init_weights) ~= length(layer_sizes)
                        error(['GAM:GAMSubunit:Constructor:', ...
                               'Invalid number of init_params'])
                    end
                elseif iscell(varargin{i+1})
                    % input is a cell array of values
                    % error-check later
                    init_weights = varargin{i+1};
                else
                    error(['GAM:GAMSubunit:Constructor:', ...
                           'Invalid init_params format']);
                end
            case 'pretraining'
                assert(ismember(varargin{i+1}, subunit.allowed_pretrainers),...
                    'GAM:GAMLayer:Constructor:Invalid pretrainer "%s"', ...
                    varargin{i+1})
                pretraining = varargin{i+1};
            otherwise
                error('GAM:GAMSubunit:Constructor:Invalid input flag "%s"', ...
                    varargin{i});
        end
        i = i + 2;
    end

    % set properties; these will be error-checked upon fitting
    subunit.add_targets = add_targets;
    subunit.mult_targets = mult_targets;
    subunit.input_target = input_target;
    subunit.input_params = input_params;
    subunit.pretraining = pretraining;
    
    % initialize array of Layer objects
    layers_(length(layer_sizes),1) = GAMLayer();
    % add layer_sizes entry for input
    layer_sizes = [prod(input_params.dims), layer_sizes];
    % add Layer objects to array
    for n = 1:length(layer_sizes)-1
        layers_(n) = GAMLayer( ...
            layer_sizes(n+1), layer_sizes(n), ...
            init_weights{n}, ... 
            'act_func', act_funcs{n});       
    end
    subunit.layers = layers_;
       
    end % method
    
end

%% ********************  setting methods **********************************
methods
    
    function subunit = set_input_params(subunit, varargin)
    % subunit = subunit.set_input_params(kargs)
    %
    % Takes a sequence of key-value pairs to set input parameters for
    % GAMSubunit object. Note that without access to the input cell array
    % there is no way to check that these changes are compatible.
    %
    % INPUTS:
    %   optional key-value pairs:
    %       'input_target', scalar
    %           input target to apply new input params to. If no 
    %           input_target is specified, defaults to 1 for all subsequent
    %           changes
    %       'dims', 1x3 vector 
    %           [num_lags, num_xpix, num_ypix]
    %           defines dimensionality of the (time-embedded) stimulus
    %       'tent_spacing', scalar
    %           optional spacing of tent-basis functions when using a 
    %           tent-basis representaiton of the stimulus. Allows for the 
    %           stimulus filters to be represented at a lower time 
    %           resolution than other model components. 
    %       'boundary_conds', 1x3 vector
    %           vector of boundary conditions on each dimension 
    %           Inf is free, 0 is tied to 0, and -1 is periodic
    %
    % OUTPUTS:
    %   subunit: updated GAMSubunit object

    % parse varargin
    assert(mod(length(varargin), 2) == 0, ...
        'GAMSubunit:set_input_params:Input should be a list of key-value pairs')
    i = 1; 
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'input_target'
                subunit.input_target = varargin{i+1};
            case 'dims'
                subunit.stim_params.dims = varargin{i+1};
                % pad stim_dims with 1's for bookkeeping
                while length(subunit.stim_params.dims) < 3
                    subunit.stim_params.dims = ...
                    cat(2, subunit.stim_params.dims, 1); 
                end
            case 'tent_spacing'
                subunit.stim_params.tent_spacing = varargin{i+1};
            case 'boundary_conds'
                assert(all(ismember(varargin{i+1}, [-1, 0, Inf])), ...
                    ['GAMSubunit:set_input_params:', ...
                     'Invalid boundary condition specified'])
                subunit.stim_params.boundary_conds = varargin{i+1};
            otherwise
                error('GAMSubunit:set_input_params:Invalid input flag "%s"', ...
                    varargin{i});
        end
        i = i + 2;
    end
   
    end % method
    
    
    function subunit = set_reg_params(subunit, varargin)
    % subunit = subunit.set_reg_params(kargs)
    %
    % Takes a sequence of key-value pairs to set regularization parameters
    % for layers in GAMSubunit object. 
    %
    % example: set reg params for all layers
    %   subunit = subunit.set_reg_params('l2_weights', 10, 'l2_biases', 1)
    % example: set reg params for specific layers
    %   subunit = subunit.set_reg_params('layers', [1, 3], 'l2_biases', 1) 
    %
    % Note: the Layer class has an equivalent method; the main usefulness
    % of this method is to quickly update the reg_params structure for ALL 
    % layers. To update single layers individually, the 
    % Layer.set_reg_params function will need to be called separately
    % for each layer
    %
    % INPUTS:
    %   optional key-value pairs:
    %       'layers', vector
    %           specify set of layers to apply the new reg_params
    %           to (default: all layers)
    %       'reg_type', scalar
    %           'l2_weights' | 'l2_biases'
    %           first input is a string specifying the type of 
    %           regularization, followed by a scalar giving the associated 
    %           regularization value, which will be applied to the layers
    %           specified by 'layers'
    %   
    % OUTPUTS:
    %   subunit: updated GAMSubunit object
    
    % define defaults
    layer_targets = 1:length(subunit.layers);
    
    % pull out 'layers' if present
    layers_loc = find(strcmp(varargin, 'layers'));
    if ~isempty(layers_loc)
        
        % verify layers_loc is valid
        assert(all(ismember(varargin{layers_loc+1}, ...
                            1:length(subunit.layers))), ...
            'GAM:GAMSubunit:set_reg_params:Invalid layers specified')
        layer_targets = varargin{layers_loc+1};
                
        % remove 'layers' from varargin; will be passed to another method
        varargin(layers_loc) = [];
        varargin(layers_loc) = []; % equiv to layers_loc+1 after deletion
        
    end
    
    % update regs
    for i = layer_targets
        subunit.layers(i) = ...
            subunit.layers(i).set_reg_params(varargin{:}); 
    end
    
    end % method
    
    
    function subunit = pretrain(subunit, Xstims, fit_layers)
    % subunit = subunit.pretrain(data)
	% 
    % Performs a layer-wise pretraining of the subunit in order to speed up
    % convergence during model fitting
    % For a symmetric network, just train encoding weights and set decoding
    % weights to be transposes
    % For a non-symmetric network, train the first half of the layers, set
    % the decoding weights in the second half to be transposes, and leave
    % the middle layer to be random weights
    %
	% INPUTS:
    %   Xstims:         cell array of * x T input matrices
    %   fit_layers:     vector of bools to denote which layers to update
    %
	% OUTPUTS:
    %   subunit:        updated GAMSubunit object
    
    % get number of units per layer
    num_layers = length(subunit.layers);
    num_nodes = zeros(num_layers + 1,1);
    for i = 1:num_layers
        num_nodes(i) = size(subunit.layers(i).weights, 2);
    end
    num_nodes(num_layers+1) = size(subunit.layers(num_layers).weights, 1);
    
    % determine properties of network
    if all(num_nodes == flipud(num_nodes))
        net_symmetric = 1;
    else
        net_symmetric = 0;
    end
    if num_layers > 1
        if mod(num_layers  -1, 2) == 0
            middle_layer = (num_layers - 1) / 2 + 1;
        else
            middle_layer = ceil((num_layers - 1) / 2);
        end
    else
        middle_layer = 0;
    end
    
    % set initial params
    temp_data = Xstims{subunit.input_target};
    
    for layer = 1:middle_layer
        
        % update params
        num_hid = num_nodes(layer+1);
        
        switch subunit.pretraining
            case 'pca'                
                temp_data = bsxfun(@minus, temp_data, mean(temp_data, 2));
                [temp_weights, ~, ~] = svd(temp_data, 'econ');
                temp_weights = temp_weights(:,1:num_hid)';
                temp_data = temp_weights * temp_data;
            case 'pca-varimax'
                temp_data = bsxfun(@minus, temp_data, mean(temp_data, 2));
                [temp_weights, ~, ~] = svd(temp_data, 'econ');
                % only rotate factors if subspace is greater than 1-d
                if num_hid > 1
                    try
                        temp_weights = rotatefactors( ...
                            temp_weights(:,1:num_hid), ...
                            'Method', 'varimax')';
                    catch ME
                        % identifier: 'stats:rotatefactors:IterationLimit'
                        try
                            temp_weights = rotatefactors( ...
                                temp_weights(:,1:num_hid), ...
                                'Method', 'varimax', ...
                                'reltol', 1e-3, ...
                                'maxit', 2000)';
                        catch ME
                            temp_weights = temp_weights(:,1:num_hid)';
                        end
                    end
                else
                    temp_weights = temp_weights(:,1)';
                end
                temp_weights = bsxfun(@times, temp_weights, ...
                                      sign(mean(temp_weights, 2)));
                temp_data = temp_weights * temp_data;
            otherwise
                error(['GAM:GAMSubunit:pretrain:', ...
                       'Invalid pretraining string "%s"'], ...
                       subunit.pretraining)
        end
        
        if fit_layers(layer) == 1
            subunit.layers(layer).weights = temp_weights;
        end
        if net_symmetric
            if fit_layers(end-layer+1) == 1
                subunit.layers(end-layer+1).weights = temp_weights';
            end
        end
        
    end
    end % method
    
end

%% ********************  getting methods **********************************
methods
    
    function [a, z] = get_model_internals(subunit, Xstims, varargin)
    % [a, z] = subunit.get_model_internals(Xstims, kargs);
    %
    % Evaluates current GAMSubunit object and returns activation values for
    % different layers of the model
    %
    % INPUTS:
    %   Xstims:     cell array of T x * stim matrices
    %
    %   optional key-value pairs:
    %       'indx_tr', vector
    %           subset of 1:T that specifies portion of data used for 
    %           evaluation (default is all data)
    %
    % OUTPUTS:
    %   z           num_layers x 1 cell array, each cell containing a
    %               matrix of the signal before being passed through
    %               the activation function of the layer
    %   a           same as z, except value of signal after being
    %               passed through the activation function

    % define defaults
    indx_tr = NaN; % NaN means we use all available data
    
    % parse varargin
    i = 1;
    while i <= length(varargin)
        switch lower(varargin{i})
            case 'indx_tr'
                assert(all(ismember(varargin{i+1}, ...
                       1:size(Xstims{subunit.input_target}, 1))), ...
                    'GAMSubunit:get_model_internals:Invalid fitting indices')
                indx_tr = varargin{i+1};
            otherwise
                error('GAMSubunit:get_model_internals:Invalid input flag "%s"', ...
                    varargin{i});
        end
        i = i + 2;
    end
   
    % use indx_tr
    input = Xstims{subunit.input_target}';
    if ~isnan(indx_tr)
        input = input(:,indx_tr);
    end
    clear Xstims % free memory
    
    % get internal generating signals
    z = cell(length(subunit.layers),1);
    a = cell(length(subunit.layers),1);
    for i = 1:length(subunit.layers)
        if i == 1
            z{i} = bsxfun(@plus, subunit.layers(i).weights*input, ...
                             subunit.layers(i).biases);
        else
            z{i} = bsxfun(@plus, subunit.layers(i).weights*a{i-1}, ...
                             subunit.layers(i).biases);
        end
        a{i} = subunit.layers(i).apply_act_func(z{i});
    end
    
    end % method
    
    
    function reg_pen = get_reg_pen(subunit)
    % reg_pen = subunit.get_reg_pen()
    %
    % Retrieves regularization penalties for each layer of GAMSubunit object
    %
    % INPUTS:
    %   none
    %
    % OUTPUTS:
    %   reg_pen: struct containing penalties due to different regs for all
    %            layers in GAMSubunit

    reg_pen = struct([]);
    for i = 1:length(subunit.layers)
        reg_pen = cat(1, reg_pen, subunit.layers(i).get_reg_pen());
    end

    end % method
    
end

end

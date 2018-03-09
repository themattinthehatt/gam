function net = fit_add_subunits(net, fs)
% net = net.fit_add_subunits(fs)
%
% Fits model parameters of additive subunits
%
% INPUTS:
%   fs:                 struct created in method GAM.fit_model
%       true_act:       num_neurons x T matrix of population activity
%       Xstims:         cell array of T x * input matrices
%
% OUTPUTS:
%   net:                updated GAM object

% ************************** DEFINE USEFUL QUANTITIES *********************
switch net.noise_dist
    case 'gauss'
        Z = numel(fs.true_act);
    case 'poiss'
        Z = sum(fs.true_act(:));
end

fit_subs = fs.fit_subs_add;
fit_layers = fs.fit_layers;
num_fit_subs = length(fit_subs);
fit_input_targs = [net.add_subunits(fit_subs).input_target];
fit_num_layers = cell(num_fit_subs, 1);
for i = 1:num_fit_subs;
    % fit_num_layers{i} is an int denoting total number of layers for each
    % subunit, including fit and unfit layers
    fit_num_layers{i} = length(net.add_subunits(fit_subs(i)).layers);
end
max_num_layers = max([fit_num_layers{:}]);

nonfit_subs = setdiff(1:length(net.add_subunits), fit_subs);

% ************************** RESHAPE WEIGHTS ******************************
init_params = [];
param_tot = 0;

% store length of each layer's weight/bias vector
sub_params = struct([]);
num_sub_params = 0;
for i = 1:num_fit_subs
    % store index values of each layer's weight/bias vector; second
    % dimension separates weights and biases
    param_lens = zeros(fit_num_layers{i}, 2);
    param_indxs_full = cell(fit_num_layers{i}, 2); % index into full param vec
    param_indxs_sub = cell(fit_num_layers{i}, 2);  % index into layer param vec
    param_tot_sub = 0;
    for j = 1:fit_num_layers{i}
        % for readability
        curr_layer = net.add_subunits(fit_subs(i)).layers(j);
        if strcmp(curr_layer.act_func, 'oneplus')
            curr_layer.biases = zeros(size(curr_layer.biases));
        end
        curr_params = [curr_layer.weights(:); ...
                       curr_layer.biases(:)];
        % add current params to initial param vector
        init_params = [init_params; curr_params];
        % get length of layer vector
        param_lens(j,1) = length(curr_layer.weights(:));
        param_lens(j,2) = length(curr_layer.biases(:));
        % param indices into full param vector
        param_indxs_full{j,1} = param_tot + (1:param_lens(j,1));
        param_tot = param_tot + param_lens(j, 1);
        param_indxs_full{j,2} = param_tot + (1:param_lens(j,2));
        param_tot = param_tot + param_lens(j, 2);
        % param indices into param vector just assoc'd w/ subunit params
        param_indxs_sub{j,1} = param_tot_sub + (1:param_lens(j,1));
        param_tot_sub = param_tot_sub + param_lens(j,1);
        param_indxs_sub{j,2} = param_tot_sub + (1:param_lens(j,2));
        param_tot_sub = param_tot_sub + param_lens(j,2);
    end
    temp_struct.param_indxs_full = param_indxs_full;
    temp_struct.param_indxs_sub = param_indxs_sub;
    temp_struct.param_tot_sub = param_tot_sub;
    
    sub_params = cat(1, sub_params, temp_struct);
    
    num_sub_params = num_sub_params + param_tot_sub;
end

% take care of final biases
curr_params = net.biases(:);
init_params = [init_params; curr_params];
num_bias_indxs = length(curr_params);
param_indxs_biases = param_tot + (1:num_bias_indxs);
param_tot = param_tot + num_bias_indxs;
    
% ************* CALCULATE NONTARGET MODEL COMPONENTS **********************
% calculate all model components
[~, ~, subs_out, ~, mult_subs_comb] = net.get_model_internals( ...
    cellfun(@(x) x', fs.Xstims, 'UniformOutput', 0));

% get combination of add/mult outputs for non-targeted additive subs
non_targ_signal = zeros(size(fs.true_act));
for i = 1:length(nonfit_subs)
    non_targ_signal = non_targ_signal + subs_out{nonfit_subs(i)}';
end

% get gain signals for targeted subs
gain_sigs = cell(num_fit_subs,1);
for i = 1:length(fit_subs)
    gain_sigs{fit_subs(i)} = mult_subs_comb{fit_subs(i)}';
end

% ************************** FIT MODEL ************************************
optim_params = net.optim_params;
if net.optim_params.deriv_check
    optim_params.Algorithm = 'quasi-newton';
    optim_params.HessUpdate = 'steepdesc';
    optim_params.GradObj = 'on';
    optim_params.DerivativeCheck = 'on';
    optim_params.optimizer = 'fminunc';
    optim_params.FinDiffType = 'central';
    optim_params.maxIter = 0;
end

% define function handle to pass to optimizer
obj_fun = @(x) objective_fun(x);

% run optimization
if strcmp(optim_params.optimizer, 'minFunc') && ~exist('minFunc', 'file')
    optim_params.optimizer = 'fminunc';
    warning('minFunc not found; switching to fminunc optimizer')
end
switch optim_params.optimizer
    case 'minFunc'
        [params, f, ~, output] = minFunc(obj_fun, init_params, ...
                                          optim_params);
    case 'fminunc'
        [params, f, ~, output] = fminunc(obj_fun, init_params, ...
                                          optim_params);
  	case 'con'
        [params, f, ~, output] = minConf_SPG(obj_fun, init_params, ...
                                          @(t,b) max(t,0), optim_params);
end

[~, grad] = objective_fun(params);
first_order_optim = max(abs(grad));
if first_order_optim > 1e-2 && optim_params.max_iter > 1
    warning('First-order optimality: %.3f, fit might not be converged!', ...
        first_order_optim);
end

% ************************** UPDATE WEIGHTS *******************************

% loop through fitted subunits
for i = 1:num_fit_subs
    for j = 1:fit_num_layers{i}
        curr_params = params(sub_params(i).param_indxs_full{j,1});
        net.add_subunits(fit_subs(i)).layers(j).weights = reshape( ...
            curr_params, size(net.add_subunits(fit_subs(i)).layers(j).weights));
        curr_params = params(sub_params(i).param_indxs_full{j,2});
        net.add_subunits(fit_subs(i)).layers(j).biases = curr_params;
    end
end

% overall biases
curr_params = params(param_indxs_biases);
net.biases = curr_params;

% ************************** UPDATE HISTORY *******************************
net = net.update_fit_history(fs, params, f, output);


    %% ******************** nested objective function *********************
    function [func, grad] = objective_fun(params)
    % Calculates the loss function and its gradient with respect to the 
    % model parameters

    % ******************* INITIALIZATION **********************************
    
    % Note: cell arrays do not need to be stored contiguously 
    z = cell(num_fit_subs, max_num_layers);
    a = cell(num_fit_subs, max_num_layers);
    weights = cell(num_fit_subs, max_num_layers);
    biases = cell(num_fit_subs, max_num_layers);
    grad_weights = cell(num_fit_subs, max_num_layers);
    grad_biases = cell(num_fit_subs, max_num_layers);
        
    % ******************* PARSE PARAMETER VECTOR **************************
    
    for ii = 1:num_fit_subs
        for jj = 1:fit_num_layers{ii}
            weights{ii,jj} = reshape( ...
                params(sub_params(ii).param_indxs_full{jj,1}), ...
                size(net.add_subunits(fit_subs(ii)).layers(jj).weights));
            biases{ii,jj} = ...
                params(sub_params(ii).param_indxs_full{jj,2});
            if strcmp(net.add_subunits(fit_subs(ii)).layers(jj).act_func, 'oneplus')
                biases{ii,jj} = zeros(length(sub_params(ii).param_indxs_full{jj,2}),1);
            end
        end
    end
    final_biases = params(param_indxs_biases);
    
    % ******************* COMPUTE FUNCTION VALUE **************************
    
    gen_sig = non_targ_signal;
    for ii = 1:num_fit_subs
        for jj = 1:fit_num_layers{ii}
            if jj == 1
                z{ii,jj} = bsxfun(@plus, weights{ii,jj} * ...
                                         fs.Xstims{fit_input_targs(ii)}, ...
                                         biases{ii,jj});
            else
                z{ii,jj} = bsxfun(@plus, weights{ii,jj} * a{ii,jj-1}, ...
                                         biases{ii,jj});
            end
            a{ii,jj} = net.add_subunits(fit_subs(ii)). ...
                           layers(jj).apply_act_func(z{ii,jj});
        end
        
        % multiply additive subunit outputs by gain
        gen_sig = gen_sig + ...
            a{ii,fit_num_layers{ii}} .* gain_sigs{fit_subs(ii)};
    end
    
    % add final bias terms
    gen_sig = bsxfun(@plus, gen_sig, final_biases);
    
    % apply spiking nonlinearity
    pred_act = net.apply_spiking_nl(gen_sig);
    
    % cost function and gradient eval wrt predicted output
    switch net.noise_dist
        case 'gauss'
            cost_grad = (pred_act - fs.true_act);
            cost_func = 0.5*sum(sum(cost_grad.^2));
        case 'poiss'
            % calculate cost function
            cost_func = -sum(sum(fs.true_act.*log(pred_act) - pred_act));
            % calculate gradient
            cost_grad = -(fs.true_act./pred_act - 1);
            % set gradient equal to zero where underflow occurs
            cost_grad(pred_act <= net.min_pred_rate) = 0;
    end
    
    % ******************* COMPUTE GRADIENTS *******************************
    
    % gradient for final biases
    Delta = net.apply_spiking_nl_deriv(gen_sig) .* cost_grad;
    param_grad = [];
    for ii = 1:num_fit_subs
        % backward pass, last layer
        delta = net.add_subunits(fit_subs(ii)).layers(end). ...
            apply_act_deriv(z{ii,fit_num_layers{ii}}) .* ...
            Delta .* gain_sigs{fit_subs(ii)};
        if fit_num_layers{ii} > 1
            % only perform if a{end-1} exists (use Xstims otherwise)
            if fit_layers{ii}(fit_num_layers{ii}) == 1
                % only perform if actually fitting
                grad_weights{ii,fit_num_layers{ii}} = ...
                    delta * a{ii,fit_num_layers{ii}-1}';
                if strcmp(net.add_subunits(fit_subs(ii)).layers(fit_num_layers{ii}).act_func, 'oneplus')
                    grad_biases{ii,fit_num_layers{ii}} = zeros( ...
                        size(biases{ii,fit_num_layers{ii}}));
                else
                    grad_biases{ii,fit_num_layers{ii}} = sum(delta, 2);
                end
            else
                grad_weights{ii,fit_num_layers{ii}} = zeros( ...
                    size(weights{ii,fit_num_layers{ii}}));
                grad_biases{ii,fit_num_layers{ii}} = zeros( ...
                    size(biases{ii,fit_num_layers{ii}}));
%                 grad_biases{ii,fit_num_layers{ii}} = sum(delta, 2);
            end
        end
        % backward pass, hidden layers
        for jj = (fit_num_layers{ii}-1):-1:2
            delta = net.add_subunits(fit_subs(ii)).layers(jj). ...
                    apply_act_deriv(z{ii,jj}) .* (weights{ii,jj+1}' * delta);
            if fit_layers{ii}(jj) == 1
                % only perform if actually fitting
                grad_weights{ii,jj} = delta * a{ii,jj-1}';
                if strcmp(net.add_subunits(fit_subs(ii)).layers(jj).act_func, 'oneplus')
                    grad_biases{ii,jj} = zeros(size(biases{ii,jj}));
                else
                    grad_biases{ii,jj} = sum(delta, 2);
                end
            else
                grad_weights{ii,jj} = zeros(size(weights{ii,jj}));
                grad_biases{ii,jj} = zeros(size(biases{ii,jj}));
%                 grad_biases{ii,jj} = sum(delta, 2);
            end
        end
        % backward pass, first hidden layer
        if fit_num_layers{ii} > 1
            delta = net.add_subunits(fit_subs(ii)).layers(1). ...
                apply_act_deriv(z{ii,1}) .* (weights{ii,2}' * delta);
            % else use delta calculated above
        end
        % backward pass, input layer
        % only perform if fitting
        if fit_layers{ii}(1) == 1
            grad_weights{ii,1} = delta * fs.Xstims{fit_input_targs(ii)}';
            if strcmp(net.add_subunits(fit_subs(ii)).layers(1).act_func, 'oneplus')
                grad_biases{ii,1} = zeros(size(biases{ii,1}));
            else
                grad_biases{ii,1} = sum(delta, 2);
            end
        else
            grad_weights{ii,1} = zeros(size(weights{ii,1}));
            grad_biases{ii,1} = zeros(size(biases{ii,1}));
%             grad_biases{ii,1} = sum(delta, 2);
        end

        % construct gradient vector
        for jj = 1:fit_num_layers{ii}
            param_grad = [ ...
                param_grad; ...
                grad_weights{ii,jj}(:); ...
                grad_biases{ii,jj}];
        end
    end
    
    bias_grad = sum(Delta, 2);
    
    % ******************* COMPUTE REG VALUES AND GRADIENTS ****************
    
    param_reg_pen = 0;
    param_reg_pen_grad = zeros(num_sub_params,1);
    for ii = 1:num_fit_subs
        for jj = 1:fit_num_layers{ii}
            % get reg pen for weights (only if fitting)
            if fit_layers{ii}(jj) == 1
                % l2
                reg_lambda_l2 = net.add_subunits(fit_subs(ii)).layers(jj). ...
                    reg_lambdas.l2_weights;
                if reg_lambda_l2 > 0
                    param_reg_pen = param_reg_pen + 0.5 * reg_lambda_l2 * ...
                        sum(weights{ii,jj}(:).^2);
                    param_reg_pen_grad(sub_params(ii).param_indxs_sub{jj,1}) = ...
                        reg_lambda_l2 * weights{ii,jj}(:);
                end
                % l1
                reg_lambda_l1 = net.add_subunits(fit_subs(ii)).layers(jj). ...
                    reg_lambdas.l1_weights;
                if reg_lambda_l1 > 0
                    param_reg_pen = param_reg_pen + reg_lambda_l1 * ...
                        sum(abs(weights{ii,jj}(:)));
                    param_reg_pen_grad(sub_params(ii).param_indxs_sub{jj,1}) = ...
                        param_reg_pen_grad(sub_params(ii).param_indxs_sub{jj,1}) + ...
                        reg_lambda_l1 * (2 * heaviside(weights{ii,jj}(:)) - 1);
                end
            end
            
            % get reg pen for biases (only if fitting)
            if fit_layers{ii}(jj) == 1
                % l2
                reg_lambda_l2 = net.add_subunits(fit_subs(ii)).layers(jj). ...
                    reg_lambdas.l2_biases;
                if reg_lambda_l2 > 0
                    param_reg_pen = param_reg_pen + 0.5 * reg_lambda_l2 * ...
                        sum(biases{ii,jj}.^2);
                    param_reg_pen_grad(sub_params(ii).param_indxs_sub{jj,2}) = ...
                        reg_lambda_l2 * biases{ii,jj};
                end
                % l1
                reg_lambda_l1 = net.add_subunits(fit_subs(ii)).layers(jj). ...
                    reg_lambdas.l1_biases;
                if reg_lambda_l1 > 0
                    param_reg_pen = param_reg_pen + reg_lambda_l1 * ...
                        sum(abs(biases{ii,jj}));
                    param_reg_pen_grad(sub_params(ii).param_indxs_sub{jj,2}) = ...
                        param_reg_pen_grad(sub_params(ii).param_indxs_sub{jj,2}) + ...
                        reg_lambda_l1 * (2 * heaviside(biases{ii,jj}) - 1);
                end
            end

        end
    end

    % regularization for final biases not currently supported
    bias_reg_pen = 0;
    bias_reg_pen_grad = zeros(size(final_biases));
    
    % ******************* COMBINE *****************************************
    
    func = cost_func / Z + ...
           param_reg_pen + bias_reg_pen;
    
    grad = [param_grad; bias_grad] / Z + ...
           [param_reg_pen_grad; bias_reg_pen_grad];
       
    end % internal function

end % fit_add_subunits method




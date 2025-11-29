    %% MARQUARDTS ALGORITM%%
    
    % 1. Enhanchment of newton's method.
    % 2. The hessian matrix is approximated by adding a factor lamda to the diagonal.
    % 3. By doing this, positve definite hessian is guaranteed. 
    % 4. Thereby the searchdirection is always a descentdirection.
    % 5. Thus, the search is not attracted to saddle points.
    
    %%
    clc; clearvars; close all;
    syms x1 x2
    
    %% Parameters %%
    x1_range = -5:1:5;  
    x2_range = -5:1:5;  
    epsilon = 10e-3;    
    t = 2;              
    lambda_factor = 15; 
    
    f_sym = -(3 * exp(-(x1^2 + x2^2)) + ...
            2 * exp(-(x1-2)^2 - (x2-2)^2) - ...
            1.5 * exp(-(x1+1.5)^2 - (x2+1.5)^2) + ...
            exp(-(x1-2)^2 - (x2+2)^2) - ...
            2 * exp(-(x1+2)^2 - (x2-2)^2));
    
    %% Pre Loop Calculations %%
    [X1, X2] = meshgrid(x1_range, x2_range);
    initial_guesses = [X1(:), X2(:)];  % Convert to list of (x1, x2) pairs
    
    grad_f_sym = gradient(f_sym, [x1 x2]);  % Gradient of f
    hess_f_sym = hessian(f_sym, [x1, x2]);  % Hessian of f
    
    % Convert symbolic expressions to function handles
    f = matlabFunction(f_sym, 'Vars', {x1, x2});
    grad_f = matlabFunction(grad_f_sym, 'Vars', {x1, x2});
    hess_f = matlabFunction(hess_f_sym, 'Vars', {x1, x2});
    
    %% Starting plot %%
    [x1_vals, x2_vals] = meshgrid(-6:0.07:6, -6:0.07:6);
    mesh_vals = f(x1_vals, x2_vals);
    
    figure;
    surf(x1_vals, x2_vals, mesh_vals);
    hold on;
    shading interp
    
    %% Tracking Performance Metrics %%
    optima = [];
    total_iterations = 0;
    
    % Timer
    tic;
    
    %% Loop Over guesses %%
    for j = 1:size(initial_guesses, 1)
        x = initial_guesses(j, :); 
        grad = grad_f(x(1), x(2)); 
        x_history = x;
        k = 0;
    
        while (epsilon < norm(grad))
    
            hess = hess_f(x(1), x(2));
            [m, n] = size(hess);
            lambda = lambda_factor * norm(grad);  
            
            % Ensure positive eigenvalues by applying dampening factor
            % This garuantees descent direction
            d = (hess + lambda * eye(n)) \ -grad;  
                
            x = x + t * d';
            x_history = [x_history; x]; 
        
            grad = grad_f(x(1), x(2));
            k = k + 1;
            total_iterations = total_iterations + 1;
        
            plot3(x(1), x(2), f(x(1), x(2)), 'bo', 'MarkerFaceColor', 'b');
        end
    
        % Store local optimum
        fval = f(x(1), x(2));
        optima = [optima; x, fval];
    
        % Plot path
        plot3(x_history(:, 1), x_history(:, 2), f(x_history(:, 1), x_history(:, 2)), 'k-', 'LineWidth', 2);
        
    end
    
    %% Results %%
    elapsed_time = toc;
    
    % Remove duplicates
    optima = unique(round(optima, 4), 'rows');
    
    % Sort optima, best minima first.
    optima = sortrows(optima, 3);
    
    %% DISPLAYING %%
    
    disp('Found local optima:');
    disp(array2table(optima, 'VariableNames', {'x1', 'x2', 'fval'}));
    
    [min_fval, min_idx] = min(optima(:,3));  
    fprintf('Global minimum: f(x1, x2) = %.6f at x1 = %.4f, x2 = %.4f\n', ...
            min_fval, optima(min_idx,1), optima(min_idx,2));
    
    fprintf('Total iterations: %d\n', total_iterations);
    fprintf('Elapsed time: %.4f seconds\n', elapsed_time);
    
    % Title and label
    title('Optimization Paths for Grid Search of Initial Points');
    xlabel('x1');
    ylabel('x2');
    zlabel('f(x1, x2)');

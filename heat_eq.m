% tic;

for kk = 0:0
    y = -1 : (2/99) : 1;
    u = (1 + y)*kk;
    N = 100;
    
    boundary_values = 0:12.5:100;

    num_conditions = numel(boundary_values)^4; % Total number of boundary conditions
    results = zeros(num_conditions,  N^2);

    counter = 1;

    for T_top = boundary_values
        for T_bottom = boundary_values
            for T_left = boundary_values
                for T_right = boundary_values

                    temperatureField = solveAdvectionDiffusion(T_top, T_bottom, T_left, T_right, kk);
    
                    temperatureVector = reshape(temperatureField, 1, []);
    
                    results(counter, :) = temperatureVector;

                    counter = counter + 1;
                end
            end
        end
    end
    
    % Write results to a CSV file
    % csv_filename = 'laplace_results.csv';
    csv_filename = 'laplace_results_v3.csv';
    
    writematrix( results, csv_filename, 'WriteMode','append');

end
% toc;
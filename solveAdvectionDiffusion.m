function temperatureField = solveAdvectionDiffusion(T_top, T_bottom, T_left, T_right, kk)
    % Solve steady-state advection-diffusion equation on a 2D domain
    % Domain: x in [0, 2], y in [-1, 1]
    % Velocity profile: u(y) = kk*(1+y)
    % Using Gauss-Seidel iteration
    
    N = 100;                       
    Lx = 2;                       
    Ly = 2;                        
    dx = Lx/(N-1);
    dy = Ly/(N-1);
    D = 1; % Diffusion coefficient

    temperatureField = zeros(N, N);
    
    % Set boundary conditions
    temperatureField(1, :) = T_top;          % Top boundary at y=1
    temperatureField(end, :) = T_bottom;     % Bottom boundary at y=-1
    temperatureField(:, 1) = T_left;         % Left boundary at x=0
    temperatureField(:, end) = T_right;      % Right boundary at x=2
    
    tolerance = 1e-1;
    maxIter = 10000;
    error = Inf;
    iter = 0;
    
    while error > tolerance && iter < maxIter
        oldTemperature = temperatureField;
        
        for i = 2:N-1
            y_i = 1 - (i-1)*dy;
            u_i = (1+y_i)*kk;
            for j = 2:N-1
                T_ip = oldTemperature(i+1, j);
                T_im = oldTemperature(i-1, j);
                T_jp = oldTemperature(i, j+1);
                T_jm = oldTemperature(i, j-1);

                temperatureField(i, j) = (T_ip + T_im + T_jp + T_jm)/4 ...
                                         - (u_i * (T_ip - T_im) * dx)/(8*D);
            end
        end
        error = max(max(abs(temperatureField - oldTemperature)));
        iter = iter + 1;
    end
end
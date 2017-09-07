x = [1.0, 1.5, 2.0, 2.5, 3.0];

BP_y = [0.24, 0.06, 0.007, 0.0008, 0.00013];
my_BP_y = [0.190493164,	0.07755, 0.00745, 0.0008251953125, 0.000155];
Ex_BP_y = [0.16, 0.05, 0.012, 0.0016, 0.00023];
my_Ex_BP_y = [0.186657715, 0.060571289, 0.011682129, 0.001699219, 0.000230469];

semilogy(x, BP_y, 'o-', x, my_BP_y, 'o-', x, Ex_BP_y, 'o-', x, my_Ex_BP_y, 'o-');
title('\alpha=1.0, \beta=0.01, \theta=5.0')
legend('BP[16]', 'result 1', 'Proposed Ex-BP', 'result 2');
xlabel('E_b/N_0 (dB)')
ylabel('BER')
grid on;


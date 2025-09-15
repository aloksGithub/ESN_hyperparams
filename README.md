
This reposiroty is meant to reproduce the results in:
"Parameterizing echo state networks for multi-step time series prediction"
https://doi.org/10.1016/j.neucom.2022.11.044

The below results are obtained by running reproduce_results.py
Run the following commands to reproduce the results for the 4 datasets
Neutral type DDE: python reproduce_results 0
Lorenz: python reproduce_results 1
MGS: python reproduce_results 2
Laser: python reproduce_results 3

Neutral type DDE washout
Best R2 over 100 seeds: 0.9999998686860551
Median R2: 0.9999987846676142
Best NRMSE: 0.1608291723449365
Median NRMSE: 1.186123708764962

Laser washout
Best R2 over 100 seeds: 0.9991013356587573
Median R2: 0.9934541515645425
Best NRMSE: 1.2411893284366495
Median NRMSE: 3.3498079153192224

MGS
Best R2 over 100 seeds: 0.9999994149551771
Median R2: 0.999961320644164
Best NRMSE: 0.012632818843692547
Median NRMSE: 0.1026138269948034
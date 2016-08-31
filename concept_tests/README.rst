Covariance matrix computation concept tests
===========================================

This section contains tests for analyzing the correctness of the covariance
matrix computation for estimated parameters from single experiments within
casiopeia.


Profit of computing covariance matrices from single experiments
---------------------------------------------------------------

The possibility to compute this matrix on the one hand allows for an
evaluation of the quality of a parameter estimation from a single data set.

Also, the matrix can be used within experimental design to optimize the
system excitation for measurement generation, and with this to improve
the information content regarding the unknown parameters of the data produced.


Test principle
--------------

During the tests, sets of pseudo-measurement data are generated repeatedly
from simulations and random normally distributed noise. For each of these
data sets, a parameter estimation is run, which leads to a set of estimated
parameter values.

Then, the standard deviations for the set of estimated parameters
generated before is compared to the standard deviations obtained from the
covariance matrix that was computed from one single experiment.

Finally, a report is generated that contains a comparison
of the standard deviations as well as the scaling factor beta.


Aim of the tests
----------------

If working correctly, the values for the standard deviations should be almost
equal. Furthermore, since the weightings applied to the 
measurements within the parameter estimations match the standard deviation
used for generating the noisy data, the scaling factor beta for the covariance
matrix should be almost 1.


Results
-------

Running the tests shows that both aims stated above are met, and the
covariance matrix computation for estimated parameters from single experiments
works as expected.

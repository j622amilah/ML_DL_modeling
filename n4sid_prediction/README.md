# n4sid_prediction

Created by Jamilah Foucher, Aout 25, 2021.

 Purpose:  Organized usage of n4sid.py, in order to test:
   1) projection coefficent methods (hankel, markov, output resized, etc), 
   2) weighting of tf (tf direct, projection coefficents), 
   3) prediction evaluation metric (r-squared, absolute error, distributed error).

Using the prediction evaluation metric, we test both discrete and continuous time usages instead of resampling the signals for one of the two constructs; oversampling past Nyquist frequency for continuous construction and exact sampling to Nyquist frequency for discrete construction.  

Thus, we decide if: (1) discrete or continuous construction is best, and (2) which of the projection coefficent methods are best.  The model with the best metic with respect to the output is returnned.

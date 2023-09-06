# Interpolation_methods

When working with time-series, it is often important to interpolate the data in order to : 1) fill in missing data points due to sampling errors, 2) make all time-series the same length for comparison of data. There are several ways to interpolate time-series: 1) stretch the signal to a certain fixed length as performed by scipy's interp1d, and 2) expand the signal by a integer multiple of the desired fixed length. Both ways give a similar result, however if you perfer to preserve the frequency of the signal precisely the 2nd method is best.

Medium blog (Practicing DatScy): https://medium.com/@j622amilah/

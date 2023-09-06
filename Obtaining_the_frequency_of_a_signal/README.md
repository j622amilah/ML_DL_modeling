# Obtaining_the_frequency_of_a_signal

Sometimes it is useful to evaluate the frequency, the periodic change, of a signal; signal frequency is called the natural frequency. Two very classical techniques are to : 1) calculate the frequency using the signal in time domain, 2) calculate the frequency using the signal in frequency domain (frequency response via the fft).

Technique 1 can be accomplished in various ways, 1) counting cycles (most simplistic), 2) fitting a sinusoid to the signal and obtaining the natural frequency of the sinusoid; within this post I practice counting cycles. Technique 2 is not often encountered in other disciplines, except control theory. The frequency is not as precise as technique 1, however for complex signals it gives a reliable estimate of the signal's natural frequency. Technique 2 is often used for human motor control because motor motion is often irregular in time domain and driven by frequency coordination.

Medium blog (Practicing DatScy): https://medium.com/@j622amilah/

# Heart_classification
Personal Biosignal Project using 'Prepared Tools for Quick direct analysis of any data: Python'

This project is inspired by wanting to learn the fundamentals of machine learning and deep learning techniques. The goal is similar to health monitoring applications (Mifit, Fitbit, etc), but using different combinations of measures that appear to influence health. More specifically, to monitor external environmental influence and physiological measures on a long-term continuous basis to find any related connections to external influences and health.

I start off by just collecting a lot of photoplethysmography (PPG) heart rate data everyday.  And, then I perform a simple analysis of seeing if there are significant trend in HR due to the time of day.  As reported in literature, there is a significant difference between daily HR while awake and sleep. Different classification methods are compared to see which method can best predict HR while awake versus sleep.

In this notebook, you will find : 
1. loading the .xls Mifit 4 data saved by the "Mi Band Notify 8.11.3" phone application, 
2. creation of the X and Y data matricies,
3. creation of X_train, Y_train, X_test, Y_test,
4. testing of several Y_train and Y_test labels (4 classes: [night_sleep=0-7h, morning=8-12h, evening=13-18h, night_awake=19-24h], 3 classes: [night_sleep=0-7h, morning=8-12h, evening=13-24h], 2 classes [night_sleep=0-7h, awake=9-24h]),
5. testing classification with a few multi-class and binary classifiers.

Medium blog (Practicing DatScy): https://medium.com/@j622amilah/

# unsupervised_label_assignment

Recently I have been exploring unsupervised labeling methods. There are many ways to construct a label for data based on different criteria : 1) feature space exploration using clustering like kmeans (which is ordering data based on the mode, because distances are chosen based on random organization. Thus, mode values of the data are the predominate clusters regardless of ordering.), 2) labeling based on outliers with respect to the data mean, 3) labeling based on the frequency of the ordered data. Similarly, for time-series or image data, measures like similarity or correlation with respect to a desired pattern (based on PCA, CCA, etc) could be used to assign labels in an unsupervised manner.

In this “practice”, I look at the first three unsupervised methods using the iris sklearn dataset.

Full post at Medium (Practicing DatScy): https://medium.com/@j622amilah/

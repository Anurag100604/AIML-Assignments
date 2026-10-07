# AIML Assignments 1-10 — Codes

These ten Python files are the final runnable versions aligned to the supplied AIML practical record (`AIML_Final.docx`). They use the same real datasets and core methods documented in the record, with the plotting/output sections included so the documented figures can be reproduced.

## Files
- 01_association_diabetes.py — Diabetes association/correlation
- 02_kmeans_clustering_wine.py — K-Means clustering on Wine
- 03_pca_wine.py — PCA on Wine
- 04_naive_bayes_breast_cancer.py — Gaussian Naive Bayes
- 05_knn_iris.py — KNN on Iris
- 06_linear_regression_grunfeld.py — Linear Regression on Grunfeld
- 07_svr_engel.py — RBF SVR on Engel
- 08_regularization_macrodata.py — Ridge/Lasso/Elastic Net on macrodata
- 09_cctv_yolov4_tiny.py — YOLOv4-tiny CCTV detection
- 10_multimodal_fake_news.py — MediaEval 2016 multimodal fake-news detection

## Notes
Assignments 1-8 load their real datasets through scikit-learn or statsmodels. Assignment 9 requires `vtest.avi`, `yolov4-tiny.cfg`, `yolov4-tiny.weights`, and `coco.names`. Assignment 10 requires `posts.txt`, `adjpg.csv`, `nadjpg.csv`, `sadjpg.csv`, and `snadjpg.csv` from the MediaEval 2016 development set.

The Assignment 10 practical record contains a documented confusion matrix that is not exactly reproduced on every current Python/scikit-learn environment. The code therefore computes the confusion matrix from the actual out-of-fold predictions and does not hard-code the reported matrix.

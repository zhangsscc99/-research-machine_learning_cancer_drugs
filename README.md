## PCA model. (and knn)
In this directory, I dealt with four sets of data(ks number at timepoint 2,3,4, and 5)
The main file is in main_folder, it is "pca_mainwork.ipynb". Conclusion is at the bottom of this file. 
In main_folder(ks4): I dealt with timepoint4 data.
In folder_ks2: I dealt with timepoint2 data.
In folder_ks3: timepoint3
In folder_ks5: timepoint5
# The problem: Combination response(Etoposide and Paclitaxel) is similar to which single drug class response at timepoint 2,3,4,and 5? Using PCA.
# Conclusion: It is similar to "Paclitaxel".

Description of how I dealt with the problem: I reorganized the data. Then I used PCA model to reduce the dimensions. (238 to 2). Then I used the component(coordinates). I made a KNN model to get the most similar single drug response. 
KNN: I used Euclidean distance. I computed the centroid of the drug combinations. I calculated the distances between the combination centroid and other single drug data points. 



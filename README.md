## ml-face-recognition

This is a face recognition tool, capable of distinguishing a person from a set of existing users. The system is tested on the AT&T database of facial images. The approached is based on three steps: 
  * (1) creating a background model
  * (2) enrolling a new user and developing an individual model based on the background knowledge 
  * (3) probing a new user by comparing the set of images with the models already enrolled in the database.

### About

This is a project that I did for the course "Fundamentals in Statistical Pattern Recognition". The implementation was
based on existing source code which provided the functionality of performing PCA on a pixel-level on the
existing images.

### Dependencies

[bob](https://www.idiap.ch/software/bob/) Machine Learning Toolbox

### Description

File | Description
------------- | -------------
project_pca_weights.py | performs feature scaling. Parameters to be set: npc, s, theta. Also fixed weights vs. varying weights
project_pca_lda.py | contains the implementation for applying LDA to the PCA projections. A different LDA model is saved for each user in addition to the PCA model. Parameters to be set: npc
project_pca_lda_weights.py | in addition to the file above, this one also uses feature scaling. Parameters to be set: npc, s, theta. Also fixed weights vs. varying weights.
project_pca_gmm.py | contains the implementation for applying GMM to the PCA projections. A different GMM model is saved for each user in addition to the PCA model. Parameters: npc, mc
project_pca_gmm_weights.py | additionally to the file above it also contains the functionality for computing the fixed and various feature scaling.
folder grid_search | contains the programs for automatically estimating the parameters. 
folder tmp | currently contains the output from running the project pca weights.py script. In case of running the variants with LDA orGMM additional files will be generated containing those models for each newly enrolled person.
file "results" | contains the results of running the grid search in the following format, on columns: number of PCA components, threshold s, boost gain theta, dev far (Development False Accept Rate), dev frr (Development False Reject Rate), eer (Equal Error Rate), test far (Test False Accept Rate), test frr (Test False Reject Rate), hter (Half-total Error Rate).


### Instructions to run 

Execute runner.py

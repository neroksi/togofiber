# Togo Fiber Optics Uptake Prediction Challenge Final Submission

This repo contains [zkiller](https://zindi.africa/users/zkiller)  solution for the ***Togo Fiber*** competition, hosted by **Zindi** on behalf of **Togo's Ministry of the Digital Economy and Transformation**. The competition focused on creating accurate and generalisable models to help the government improve connectivity, allocate resources efficiently, and foster digital inclusivity.

Technically, this task can be carried out as tabular-data-classification problem.


**Notes:** This script [can be run on Kaggle](https://www.kaggle.com/code/kneroma/zindi-togo-fiber/notebook).

### 0. General notes
* All path specification is either absolute, either relative the project root folder
* The main script is in ``togofiber/main.py``
* It's highly recommended to execute the scripts from  the project root folder

The folders tree should look like (see ``tree.txt`` for full tree):

````
C:.
¦   .gitignore
¦   readme.md
¦   requirements.txt
¦   setup.py
¦   tree.txt
¦
+---data
¦       cluster_dict_aggclust_240517.json
¦       fold_dict_skf_aggclust_240517.json
¦       submission.csv
¦       sub_catb_*.csv
¦       sub_lgbm_*.csv
¦       Test.csv
¦       Train_Full.csv
¦
+---models
¦       catb_*.bin
¦       lgbm_*.bin
¦
+---src
    +---togofiber
    ¦   bagging.py
    ¦   config.py
    ¦   dataset.py
    ¦   knn_feats.py
    ¦   main.py
    ¦   training.py
    ¦   train_catboost.py
    ¦   train_lgbm.py
    ¦   utils.py
    ¦   __init__.py
````

### 1. Python 3.10.12 Installation
We run all our experiments under ***Python 3.10.12***. We highly recommend to run the scripts under this same settings even if they could (hopefully) run under different environments.

To install this project and all its requirements, run the bellow snippet from the project root folder:

````bash
pip install -e .
````

This will install all the requirements. Note that you need to install Python 3.10.12 before running the above command.

### 2. Hardware requirements
All the experiments have been conducted under python version 3.10.12. We mainly train the models on a local laptop (8 cores, 16 Gb RAM). A typical training session would last around 15 mins and inference is almost instataneous.

### 3. Model Train & Inference
All the script needed to make a prediction is in ``togofiber/main.py``. The `main()` function is surely what you need.

The easieast way to make this script work is to follow the proposed folder structure (see `tree.txt`), otherwise one will need to adjust some path variable under the `togofiber/config.py` module.

For a fast demo run, set ``IS_DEBUG`` to ***True*** in ``togofiber`/config.py``, just set it to ***False*** for *train + inference* on the whole data set. For the whole data set, the train+inference script should take around 20 minutes. This require validation  folds to be already created, otherwise see **section 5** for folds creation.

````bash
python togofiber/main.py
````
The above script will read all the required files (`Train_Full.csv`, `Test.csv`, `fold_dict_skf_aggclust_240517.json`, `cluster_dict_aggclust_240517.json`), prepare the data and lunch training + inference session. The final predictions will be stored under `<SUB_SAVE_ROOT>/submission.csv`. Note that the `folder_dict` and `cluster_dict` files are meant for k-fold validation splitting and are provided by us.

For more details on the modeling approch, see the next session.

### 4. Modeling Details
Training is solely based on the provided data, no external data were used. We approach this task as a standard tabular-data-classification problem, hence we used these stacks:

* **validation** : kfold stratified validation based on custom computed clusters (please see `togofiber/dataset.py:build_clusters_and_folds()` or the bellow section)
* **feature engineering**: we systematically use OneHotEncoding for categorical features, standardize variables based on their logical-group. But our most important features are those from KNN clustering. At the early beginning of this competition we were surprised by the good results of a simple KNN model (~0.90 AUC). Hence we use KNN to retrieve 81 neighbours for each data point and then use the averged neighbors-features for our final boosting models.
* **modeling** : we solely use boosting models (LightGBM & CatBoost) as one can see under `togofiber/training.py`. The best features are from KNN-neighborhood.
* **ensembling** : we used a dummy average ensembling technique with same weight (0.5) for both models

Let recall that apart from those things mentionned above, we tried several ideas with little or no success:

* Label encoding
* Pseudo Labelling
* Custom Transformer Neural Network (quickly discarded given the poor results)


### 5. Building validation folds
As stated above, we used kfold stratified validation based on custom computed clusters. This can be done running:

````python
>>> from togofiber.dataset import build_clusters_and_folds
>>> cluster_dict, fold_dict = build_clusters_and_folds(debug=False, save=True) 
````

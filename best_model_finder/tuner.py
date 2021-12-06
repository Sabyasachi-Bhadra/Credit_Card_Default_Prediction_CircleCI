from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score,accuracy_score


class Model_Finder:
    """
    This class shall be used to find the model with bst accuracy and AUC score.
    Written By : Sabyasachi
    Version : 1.0
    Revisions : None
    """

    def __init__(self, file_object, logger_obhect):
        self.file_object = file_object
        self.logger_object = logger_obhect
        self.rf_clf = RandomForestClassifier()
        self.xgb_clf = XGBClassifier(objective="binary:logistic")

    def get_best_params_for_random_forest(self, train_x, train_y):
        """
        Method Name: get_best_params_for_random_forest
        Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                     Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: Sabyasachi
        Version: 1.0
        Revisions: None
        """
        self.logger_object.log(self.file_object, "Entered the get_best_params_for_random_forest_method "
                                                 "of the Model_Finder class")
        try:
            #  initializing with different combination of RandomForest parameters
            self.param_grid_rf = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                               "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}
            #  creating an object of the grid search class
            self.grid = GridSearchCV(estimator=self.rf_clf, param_grid=self.param_grid_rf, cv=5, verbose=3)
            #  finding the best parameters
            self.grid.fit(train_x, train_y)
            #  extracting the best parameters
            self.criterion = self.grid.best_params_["criterion"]
            self.max_depth = self.grid.best_params_["max_depth"]
            self.max_features = self.grid.best_params_["max_features"]
            self.n_estimators = self.grid.best_params_["n_estimators"]

            #  creating a new model with the best parameters
            self.rf_clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                                 max_depth=self.max_depth, max_features=self.max_features)
            #  training the new model
            self.rf_clf.fit(train_x, train_y)
            self.logger_object.log(self.file_object, "RandomForest best params : %s. "
                                                     "Exited the get_best_params_for_random_forest "
                                                     "method of the Model_Finder class" % self.grid.best_params_)
            return self.rf_clf
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in get_best_params_for_random_forest "
                                                     "method of the Model_Finder class. Exception Message : %s" % e)
            self.logger_object.log(self.file_object, "RandomForest parameter tuning failed. "
                                                     "Exited the get_best_params_for_random_forest method "
                                                     "of the Model_Finder class")
            raise e

    def get_best_params_for_xgboost(self, train_x, train_y):
        """
        Method Name: get_best_params_for_xgboost
        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                     Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: Sabyasachi
        Version: 1.0
        Revisions: None

        """
        self.logger_object.log(self.file_object, "Entered the get_best_params_xgboost method "
                                                 "of the Model_Finder class")
        try:
            #  initializing with different combination of XGBoost parameters
            self.param_grid_xgboost = {'learning_rate': [0.5, 0.1, 0.01, 0.001],
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [10, 50, 100, 200]
                                       }
            #  creating an object of the Grid Search CV
            self.grid = GridSearchCV(estimator=self.xgb_clf, param_grid=self.param_grid_xgboost, cv=5, verbose=3)
            #  finding the best parameters
            self.grid.fit(train_x, train_y)
            #  extracting the best parameters
            self.learning_rate = self.grid.best_params_["learning_rate"]
            self.max_depth = self.grid.best_params_["max_depth"]
            self.n_estimators = self.grid.best_params_["n_estimators"]

            #  creating a new model with the best parameters
            self.xgb_clf = XGBClassifier(learning_rate=self.learning_rate, max_depth=self.max_depth,
                                         n_estimators=self.n_estimators)

            # training the new model
            self.xgb_clf.fit(train_x, train_y)
            self.logger_object.log(self.file_object, "XGBoost best params : %s. "
                                                     "Exited get_best_params_for_xgboost method "
                                                     "of the Model_Finder class" % self.grid.best_params_)
            return self.xgb_clf
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in get_best_params_for_xgboost method "
                                                     "of the Model_Finder class. Exception Message : %s" % e)
            self.logger_object.log(self.file_object, "XGBoost parameter tuning failed. "
                                                     "Exited the get_best_params_for_xgboost method "
                                                     "of the Model_Finder class")
            raise e

    def get_best_model(self, train_x, train_y, test_x, test_y):
        """
        Method Name: get_best_model
        Description: Find out the Model which has the best AUC score.
        Output: The best model name and the model object
        On Failure: Raise Exception

        Written By: Sabyasachi
        Version: 1.0
        Revisions: None

        """
        self.logger_object.log(self.file_object, "Entered the get_best_model method"
                                                 " of the Model_Finder class")

        try:
            # create the best model for XGBoost
            self.xgboost = self.get_best_params_for_xgboost(train_x, train_y)
            self.prediction_xgboost = self.xgboost.predict(test_x)  #  prediction using XGBoost model

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
                self.logger_object.log(self.file_object, "Accuracy for XGBoost : %s" % self.xgboost_score)
            else:
                self.xgboost_score =roc_auc_score(test_y, self.prediction_xgboost)
                self.logger_object.log(self.file_object, "AUC for XGBoost : %s" % self.xgboost_score)

            # create the best model for RandomForest
            self.random_forest = self.get_best_params_for_random_forest(train_x, train_y)
            self.prediction_random_forest = self.random_forest.predict(test_x)  # prediction using RandomForest model

            if len(test_y.unique()) == 1: #  if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.random_forest_score = accuracy_score(test_y, self.prediction_random_forest)
                self.logger_object.log(self.file_object, "Accuracy score for RandomForest : %s" % self.random_forest_score)
            else:
                self.random_forest_score = roc_auc_score(test_y, self.prediction_random_forest)
                self.logger_object.log(self.file_object, "AUC for RandomForest : %s " % self.random_forest_score)

            # comparing two models
            if self.random_forest_score < self.xgboost_score:
                return "XGBoost", self.xgboost
            else:
                return "RandomForest", self.random_forest
        except Exception as e:
            self.logger_object.log(self.file_object, "Exception occurred in get_best_model method "
                                                     "of the Model_Finder class. exception Message : %s" % e)
            self.logger_object.log(self.file_object, "Model selection failed. "
                                                     "Exited the get_best_model of the Model_Finder class")
            raise e






















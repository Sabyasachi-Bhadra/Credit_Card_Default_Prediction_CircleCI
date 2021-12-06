import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

class Preprocessor:
    """
    This class shall be used to clean and transform the data before training
    Written By : Sabyasachi
    Version : 1.0
    Revisions : None

    """
    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object

    def remove_columns(self,data,columns):
        """
                Method Name: remove_columns
                Description: This method removes the given columns from a pandas dataframe.
                Output: A pandas DataFrame after removing the specified columns.
                On Failure: Raise Exception

                Written By: Sabyasachi
                Version: 1.0
                Revisions: None

        """
        self.logger_object.log(self.file_object,"Entered the remove_columns method of the Preprocessor class")
        self.data = data
        self.columns = columns
        try:
            self.useful_data = self.data.drop(labels=self.columns,axis=1) # drop the labels specified in the columns
            self.logger_object.log(self.file_object,
                                   "Column removal Successful.Exited the"
                                   " remove_columns method of the Preprocessor class")
            return self.useful_data
        except Exception as e:
            self.logger_object.log(self.file_object,"Exception occurred in remove_columns"
                                                    " method of the Preprocessor class. Exception message:  '+str(e)")
            raise e

    def separate_label_feature(self,data,label_column_name):
        """
        Method Name: separate_label_feature
        Description: This method separates the features and a Label Column
        Output: Returns two separate Dataframes, one containing features and the other containing Labels .
        On Failure: Raise Exception

        Written By: Sabyasachi
        Version: 1.0
        Revisions: None

        """
        self.logger_object.log(self.file_object,'Entered the separate_label_feature method of the Preprocessor class')
        try:
            self.X = data.drop(labels=label_column_name,axis=1) # drop the columns specified and separate the feature columns
            self.Y = data[label_column_name] # Filter the Label column
            self.logger_object.log(self.file_object,"Label Separation Successful. "
                                                    "Exited the separate_label_feature method of the Preprocessor class")
            return self.X, self.Y
        except Exception as e:
            self.logger_object.log(self.file_object,"Exception occurred in separate_label_feature method "
                                                    "of the Preprocessor class. Exception message : ' + str(e)")
            raise e

    def is_null_present(self,data):
        """
        Method Name: is_null_present
        Description: This method checks whether there are null values present in the pandas Dataframe or not.
        Output: Returns a Boolean Value. True if null values are present in the DataFrame, False if they are not present.
        On Failure: Raise Exception

        Written By: Sabyasachi
        Version: 1.0
        Revisions: None

        """
        self.logger_object.log(self.file_object,"Entered the is_null_present method of the Preprocessor class")
        self.null_present = False
        try:
            self.null_counts = data.isna().sum() # check for the count of null values per column
            for i in self.null_counts:
                if i > 0:
                    self.null_present = True
                    break
            if self.null_present: # write the logs to see which columns have null values
                dataframe_with_null = pd.DataFrame()
                dataframe_with_null['columns'] = data.columns
                dataframe_with_null['missing_value_count'] = np.asarray(data.isna().sum())
                dataframe_with_null.to_csv('preprocessing_data/null_values.csv') # storing the null column information to file
            return self.null_present
        except Exception as e:
            self.logger_object.log(self.file_object,"Exception occurred in is_null_present "
                                                    "method of the Preprocessor class. Exception Message : " + str(e))
            raise e

    def impute_missing_values(self,data):
        """
        Method Name: impute_missing_values
        Description: This method replaces all the missing values in the Dataframe using KNN Imputer.
        Output: A Dataframe which has all the missing values imputed.
        On Failure: Raise Exception

        Written By: Sabyasachi
        Version: 1.0
        Revisions: None
        """
        self.logger_object.log(self.file_object,"Entered the impute_missing_values method of the Preprocessor class")
        self.data = data
        try:
            imputer = KNNImputer(n_neighbors=3,weights='uniform',missing_values=np.nan)
            self.new_array = imputer.fit_transform(self.data) # impute the missing values
            # convert the nd-array returned in the step above to a Dataframe
            self.new_data = pd.DataFrame(data=self.new_array , columns=self.data.columns)
            self.logger_object.log(self.file_object,"Imputing missing values Successful. "
                                                    "Exited the impute_missing_values method of the Preprocessor class")
            return self.new_data
        except Exception as e:
            self.logger_object.log(self.file_object,"Exception occurred in impute_missing_values "
                                                    "method of the Preprocessor class. Exception message: ' + str(e)")
            raise e

    def get_columns_with_zero_std_deviation(self,data):
        """
        Method Name: get_columns_with_zero_std_deviation
        Description: This method finds out the columns which have a standard deviation of zero.
        Output: List of the columns with standard deviation of zero
        On Failure: Raise Exception

        Written By: Sabyasachi
        Version: 1.0
        Revisions: None
        """
        self.logger_object.log(self.file_object,"Entered the get_columns_with_zero_std_deviation "
                                                "method of the Preprocessor class")
        self.columns = data.columns
        self.data_n = data.describe()
        self.col_to_drop = []
        try:
            for column in self.columns:
                if self.data_n[column]['std'] == 0: # check if standard deviation is zero
                    self.col_to_drop.append(column) # prepare the list of columns with standard deviation zero
            self.logger_object.log(self.file_object,"Column search for Standard Deviation of Zero Successful. "
                                                    "Exited the get_columns_with_zero_std_deviation "
                                                    "method of the Preprocessor class")
            return self.col_to_drop
        except Exception as e:
            self.logger_object.log(self.file_object,"Exception occured in get_columns_with_zero_std_deviation "
                                                    "method of the Preprocessor class. Exception message:  ' + str(e)")
            raise e


    def normalization_of_data(self,data):
        """
        Method Name : normalization_of _data
        Description : This method will scale down the dataset.
        Output : Scaled Data
        On Failure : Raise Exception
        Written By : Sabyasachi
        Version : 1.0
        Revisions : None
        """
        self.logger_object.log(self.file_object,"Normalization of the data started of the preprocessor class")
        self.data = data
        try:
            scalar = StandardScaler()
            self.new_array = scalar.fit_transform(self.data)
            # convert the nd-array returned in the step above to a Dataframe
            self.new_data = pd.DataFrame(data=self.data,columns=self.data.columns)
            self.logger_object.log(self.file_object, "Normalization of data Successful."
                                                 "Exited the normalization_of_data method of the Preprocessor class")
            return self.new_data
        except Exception as e:
            self.logger_object.log(self.file_object,"Exception occurred in normalization_of_data"
                                                    "method of the Preprocessor class. Exception message: ' + str(e)")
            raise e









































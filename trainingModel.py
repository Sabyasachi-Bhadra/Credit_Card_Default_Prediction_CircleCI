"""
This is the Entry point for Training the Machine Learning Model.

Written By: Sabyasachi
Version: 1.0
Revisions: None

"""

# Doing the necessary import
from sklearn.model_selection import train_test_split
from data_ingestion.data_loader import Data_Getter
from data_preprocessing.preprocessing import Preprocessor
from data_preprocessing.clustering import KMeansClustering
from best_model_finder.tuner import Model_Finder
from file_operations.file_methods import File_Operation


from application_logging.logger import App_Loger

class trainModel:
    def __init__(self):
        self.log_writer = App_Loger()
        self.file_object = open("Training_Logs/ModelTrainingLog.txt","a+")

    def trainingModel(self):
        self.log_writer.log(self.file_object,"start of Training")
        try:
            # Getting the data from the source
            data_getter = Data_Getter(self.file_object,self.log_writer)
            data = data_getter.get_data()

            preprocessing = Preprocessor(self.file_object,self.log_writer)
            data = preprocessing.remove_columns(data,'ID') # remove the ID column as it doesn't contribute to prediction.
            # create separate features and labels
            X,Y = preprocessing.separate_label_feature(data,label_column_name='default.payment.next.month')

            # check if missing values are present in the dataset
            is_null_present = preprocessing.is_null_present(X)

            # if missing values are there, replace them appropriately.
            if is_null_present:
                X = preprocessing.impute_missing_values(X) # missing value imputation

            # check further which columns do not contribute to predictions
            # if the standard deviation for a column is zero, it means that the column has constant values
            # and they are giving the same output both for good and bad sensors
            # prepare the list of such columns to drop

            cols_to_drop = preprocessing.get_columns_with_zero_std_deviation(X)

            # drop the columns obtained above
            X = preprocessing.remove_columns(X,cols_to_drop)

            # Normalization
            X = preprocessing.normalization_of_data(X)

            """Applying the Clustering approach"""
            kmeans = KMeansClustering(self.file_object, self.log_writer)  # object initialization
            number_of_clusters = kmeans.elbow_plot(X)  # using the elbow plot to find the number of optimum clusters
            # Divide the data into Clusters
            X = kmeans.create_clusters(X, number_of_clusters)

            # create a new column in the dataset consisting of the corresponding cluster assignments.
            X['Labels'] = Y

            # Getting the unique clusters from our data set
            list_of_clusters = X['Cluster'].unique()

            """parsing all the clusters and looking for the best ML algorithm to fit on individual cluster"""

            for i in list_of_clusters:
                cluster_data = X[X['Cluster']==i]  # filter data for one cluster

                # prepare the feature and label column
                cluster_features = cluster_data.drop(['Labels', 'Cluster'], axis=1)
                cluster_label = cluster_data['Labels']

                # splitting the data into training and test set for each cluster one by one
                x_train, x_test, y_train, y_test = train_test_split(cluster_features, cluster_label, test_size=1/3, random_state=355)

                model_finder = Model_Finder(self.file_object, self.log_writer)  # object initialization

                # getting the best model for each clusters
                best_model_name, best_model = model_finder.get_best_model(x_train, y_train, x_test, y_test)

                # saving the best model to the directory
                file_op = File_Operation(self.file_object, self.log_writer)  # object initialization
                save_model = file_op.save_model(best_model, best_model_name+str(i))

            self.log_writer.log(self.file_object, "End of Training : Successful")
            self.file_object.close()

        except Exception:
            self.log_writer.log(self.file_object, "End of Training : Unsuccessful ")
            self.file_object.close()
            raise Exception

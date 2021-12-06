from datetime import datetime
from os import listdir
import pandas
from application_logging.logger import App_Loger

class dataTranform:

    """
    This class shall be used for transforming the good raw training data before loading it in Database.
    Written By : Sabyasachi
    Version : 1.0
    Revisions : None
    """
    def __init__(self):
        self.goodDataPath = "Training_Raw_files_validated/Good_Raw"
        self.logger = App_Loger()

    def replaceMissingWithNull(self):
        """
        Method Name: replaceMissingWithNull
        Description: This method replaces the missing values in columns with "NULL" to
                     store in the table. We are using substring in the first column to
                     keep only "Integer" data for ease up the loading.
                     This column is anyways going to be removed during training.

        Written By: Sabyasachi
        Version: 1.0
        Revisions: None
        """
        log_file = open("Training_Logs/dataTransformLog.txt", "a+")
        try:
            onlyfiles = [f for f in listdir(self.goodDataPath)]
            for file in onlyfiles:
                csv = pandas.read_csv(self.goodDataPath+'/'+file)
                csv.fillna('NULL',inplace=True)
                csv.to_csv(self.goodDataPath+'/'+file,index=None,header=True)
                self.logger.log(log_file,"%s : File Transformed successfully!!" % file)

        except Exception as e:
            self.logger.log(log_file,"Data Transformation failed!! :: %s" %e)
            log_file.close()

        log_file.close()









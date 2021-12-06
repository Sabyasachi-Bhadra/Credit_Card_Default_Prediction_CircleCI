import pandas as pd
# from application_logging.logger import App_Loger


class Data_Getter:
    """
    This class shall be used for obtaining the data from the source of training

    Written By : Sudhanshu & Sabyasachi
    Version : 1.0
    Revisions : None
    """

    def __init__(self,file_object, logger_object):
        self.training_file = "Training_FileFromDB/InputFile.csv"
        self.file_object = file_object
        self.logger_object = logger_object

    def get_data(self):
        """
        Method Name : get_data
        Description : This method read the data from the source
        Output : A pandas DataFrame
        On Failure : Raise Exception

        Written By : Sudhanshu & Sabyasachi
        Version : 1.0
        Revisions : None
        """
        self.logger_object.log(self.file_object, "Entered the get_data method of the Data_Getter class")
        try:
            data = pd.read_csv(self.training_file) # reading the data file
            self.logger_object.log(self.file_object,
                                   "Data Load Successfully. "
                                   "Exited the get_data method of the Data_Getter class")

            return data
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   "Exception occurred in get_data method of the Data_Getter class. "
                                   "Exception message : ' + str(e)")
            raise Exception()













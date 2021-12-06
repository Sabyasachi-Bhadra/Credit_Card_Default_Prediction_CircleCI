from Training_Raw_data_validation.rawValidation import Raw_Data_validation
from DataTypeValidation_Insertion_Training.DataTypeValidation import dBOperation
from DataTransform_Training.DataTransformation import dataTranform
from application_logging import logger

class train_validation:
    def __init__(self,path):
        self.raw_data = Raw_Data_validation(path)
        self.dataTransform = dataTranform()
        self.dBOperation = dBOperation()
        self.file_object = open("Training_Logs/Training_Main_Log.txt", "a+")
        self.log_writer = logger.App_Loger()

    def train_validation(self):
        try:
            self.log_writer.log(self.file_object, "Start of Validation on files!!")
            # extracting values from training schema
            Name_Pattern_0, Name_Pattern_1, Name_Pattern_2, column_names, noofcolumns = self.raw_data.valuesFromSchema()
            # validating filename of prediction files
            self.raw_data.validationonFileNameRaw(Name_Pattern_0, Name_Pattern_1, Name_Pattern_2)

            # validating column length in the file
            self.raw_data.validateColumnLength(noofcolumns)
            # validating if any column has all values missing
            self.raw_data.validateMissingValuesInWholeColumn()
            self.log_writer.log(self.file_object, "Raw Data Validation Completed!!")

            self.log_writer.log(self.file_object, "Starting Data Transformation!!")
            self.dataTransform.replaceMissingWithNull()

            self.log_writer.log(self.file_object, "DataTrasformation Completed!!")

            self.log_writer.log(self.file_object,
                                "Creating Training_Database and tables on the basis"
                                "of given schema"
                                )
            # create database with given name, if present open the connection!
            # Create table with columns given in schema
            self.dBOperation.createTableDb('Training',column_names)
            self.log_writer.log(self.file_object, "Table Creation Completed!!")
            self.log_writer.log(self.file_object, "Insertion of Data into Table started!!")
            # insert csv files in the table
            self.dBOperation.insertIntoTableGoodData('Training')
            self.log_writer.log(self.file_object, "Insertion into Table completed!!")
            self.log_writer.log(self.file_object, "Deleting Good Data Folder!!")
            # Delete the good data folder after loading files in table
            self.raw_data.deleteExistingGoodDataTrainingFolder()
            self.log_writer.log(self.file_object, "Good_Data folder deleted!!")
            self.log_writer.log(self.file_object, "Moving Bad files to Archive and deleting Bad_Data folder!!")
            # Move the bad files to archive folder
            self.raw_data.moveBadFilesToArchiveBad()
            self.log_writer.log(self.file_object, "Bad files moved to archive!! Bad folder Deleted!!")
            self.log_writer.log(self.file_object, "Validation Operation Completed!!")
            self.log_writer.log(self.file_object, "Extracting csv file from table")
            # export data in table to csv file
            self.dBOperation.selectingDatafromtableintocsv("Training")
            self.file_object.close()

        except Exception as e:
            raise e






















































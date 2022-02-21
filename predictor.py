# Import Necessary Libraries
import os 
import gcsfs
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


# Create Class for model prediction
class PetFinderClassifier:
    
    def __init__(self):
        """init constructor"""
        self.data = None
        self.features = None
        self.labels = None
        self.predictor = None

    
    def load_data(self, path):
        """data loader method"""
        fs = gcsfs.GCSFileSystem(project='XGBoost_Classifier')

        with fs.open(path) as f:
            self.data = pd.read_csv(f)
            
            # Split Data into Training Features and Label
            self.features = self.data.iloc[:, :-1]
            self.labels = self.data.iloc[:, -1]
            
            # Convert training features from sting to category
            self.features = self.features.astype("category")

            # Encode Label column from string to integers
            label_encoder = LabelEncoder()
            label_encoder = label_encoder.fit(self.labels)
            self.labels = label_encoder.transform(self.labels)
        
        return self.data, self.features, self.labels, label_encoder
        
    
    
    def load_model_from_disk(self, path):
        """load model from disk """
        self.new_model = XGBClassifier()
        self.new_model.load_model(path)
        return self.new_model
    
    
    
    def prediction(self, path):
        """ predict data on trained model """
        self.new_data, self.new_features, self.new_labels, label_encoder = self.load_data(path)
        
        save_data = self.new_data.copy(deep=True)

        best_tree = self.new_model.best_ntree_limit
        
        y_pred = self.new_model.predict(self.new_features, iteration_range=(0,best_tree))

        self.predictor = pd.DataFrame(y_pred, columns=["Adopted_prediction"])

        
        self.new_data['Adopted_prediction'] = self.predictor['Adopted_prediction'].apply(lambda x: 'No' if x == 0 else 'Yes')        
        
        if not os.path.exists("output"):
            os.makedirs("output") 

        self.new_data.to_csv("output/results.csv")

        return save_data, self.new_features, self.new_data, label_encoder
      


# create classifier object
Classifier = PetFinderClassifier()


# Load trained model from disk
path = "artifacts/model/xgboost_classifier.json"
Classifier.load_model_from_disk(path)


# Load data from google cloud
path = "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"
real_data, features, result, label_encoder = Classifier.prediction(path) 


#  print new result
print(result.head())



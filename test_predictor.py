# Import Necessary Libraries
import unittest
from predictor import PetFinderClassifier


# Create Unit Test Class
class TestInt(unittest.TestCase):
    
    def setUp(self):
        """create global variable for unit test"""
        self.Classifier = PetFinderClassifier()


        # Load trained model from disk
        path = "artifacts/model/xgboost_classifier.json"
        self.Classifier.load_model_from_disk(path)
    
        # Load data from google cloud
        path = "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"
        self.real_data, self.features, self.result, self.le = self.Classifier.prediction(path)
    
    
    
    def test_inputData(self):
        """ Check input data dimension """
        self.assertEqual((11537, 14), self.real_data.shape)
  
      
    def test_inputFeatures(self):
        """ Check if training features exclude label """
        self.assertEqual((11537, 13), self.features.shape)
        
    def test_resultDimension(self):
        """ Check if final output result includes Adopted prediction column """
        self.assertEqual((11537, 15), self.result.shape)
        
        
    def test_predictorProbability(self):
        """ Check predicted probablity values are either 1 or 0 """
        #proba = list(self.Classifier.predictor['Adopted_prediction'].unique())
        value = list(self.le.transform(['No', 'Yes']))
        self.assertEqual(value, [0, 1])
        
        
    def test_predictorValue(self):
        """ Check predicted classes are either YES or NO """
        #values = list(self.result['Adopted_prediction'].unique())
        classes = list(self.le.classes_)
        self.assertEqual(classes, ['No', 'Yes'])        




# Call unittest class
unittest.main(argv=[''], verbosity=2, exit=False)




# Import Libraries
import pandas as pd
import numpy as np
from copy import deepcopy
#from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

class ForwardStepwise:
    def __init__(self, dataset, model_type, target_feature, scoring):
        self.dataset = dataset
        self.target_feature = dataset[target_feature]

        self.dataset.drop(columns=[target_feature], inplace=True)

        self.features = [[feature] for feature in dataset.columns]
        self.model_type = model_type
        
        self.scoring = scoring

    def getXY(self, feature):
        #y = self.dataset[self.target_feature]
        x = self.dataset[feature]

        return x
    
    def trainModel(self, x, y):
        model = self.model_type.fit(x, y)
        return model
    
    def evaluateModel(self, model, x, y):
        metric = np.mean(cross_val_score(model, x, (y), cv=10, scoring=self.scoring))
        return metric
    
    def forwardStepwise(self):
        feature0 = self.features[0]
        #self.features.remove(feature0)
        x0 = self.getXY(feature0)

        best_features = feature0.copy()
        best_model = self.trainModel(x0, self.target_feature)
        best_metric = self.evaluateModel(best_model, x0, self.target_feature)

        #predictors = [[feature] for feature in self.features]

        while True:
            improved = False

            for feature in self.features:
                x = self.getXY(feature)
                model = self.trainModel(x, self.target_feature)
                mean_metric = self.evaluateModel(model, x, self.target_feature)

                if mean_metric > best_metric:
                    improved = True
                    best_features = feature.copy()
                    best_model = deepcopy(model)
                    best_metric = mean_metric.copy()
                else:
                    pass

                print(f'Best Features: {best_features}, Best {self.scoring}: {best_metric}, Comparison Feature: {feature}, Comparison {self.scoring}: {mean_metric}')
            
            new_features = []
            for feature in self.dataset.columns:
                if feature not in best_features:
                    new_features.append([feature] + best_features)
            
            self.features = new_features.copy()
            #best_features = self.features[0]
            #self.features = [best_features.copy().append(predictor) if predictor not in self.features[0] else best_features.copy() for predictor in self.dataset.columns]

            if improved == False:
                break

        return best_model, best_features, best_metric



import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression


from sklearn.ensemble import StackingClassifier, StackingRegressor


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report


from xgboost import XGBClassifier, XGBRegressor


from lightgbm import LGBMClassifier, LGBMRegressor

from catboost import CatBoostClassifier, CatBoostRegressor


import warnings
warnings.filterwarnings("ignore")
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            rf_best = RandomForestClassifier(
                n_estimators=150,       
                max_depth=10,          
                min_samples_split=5,     
                min_samples_leaf=5,     
                max_features='sqrt',     
                bootstrap=False,
                class_weight='balanced',
                random_state=42
            )


            lgb_best = LGBMClassifier(
                n_estimators=800,        
                learning_rate=0.0110008904745473,      
                max_depth=10,            
                num_leaves=84,          
                min_child_samples=8,
                subsample= 0.7662986173909847,           
                colsample_bytree=0.5361408003314709,    
                reg_lambda=4.202883246449607,           
                reg_alpha=2.471782553106263,             
                objective='binary',
                random_state=42
            )


            cat_best = CatBoostClassifier(
                iterations=800,          
                learning_rate=0.01389054425822741,     
                depth=7,               
                l2_leaf_reg=6.787941157323723,          
                loss_function='Logloss',
                subsample=0.6023248935827203,
                border_count=195,
                eval_metric='AUC',
                random_seed=42,
                verbose=0
            )


            xgb_best = XGBClassifier(
                n_estimators=500,         
                learning_rate=0.02955601818454443,     
                max_depth=7,             
                subsample=0.9321324446331001,           
                colsample_bytree=0.6320929337696162,    
                gamma=1.739388933239252,              
                reg_lambda=2.1987894133144046,           
                reg_alpha=0.16543161634688805,             
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=42,
                use_label_encoder=False,
                verbose=0
            )




            estimators = [
            ('rf', rf_best),
            ('lgbm', lgb_best),
            ('xgb', xgb_best),
            ('cat', cat_best)
            ]

            meta_model = LogisticRegression(max_iter=500)

            stack = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_model,
                cv=5,
                n_jobs=-1
            )
            
            stack.fit(X_train, y_train)

            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=stack
            )


            y_pred = stack.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            logging.info(f"Stacked model trained successfully | Accuracy: {accuracy}")
            



            
        except Exception as e:
            raise CustomException(e,sys)
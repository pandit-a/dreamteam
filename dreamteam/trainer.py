import joblib
from getdata import get_data, clean_data, preproc_gen, preproc_x, preproc_y
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score

##### trainer.py ####


class Trainer(object):
    def __init__(self, X, y):
        """
            X: pandas DataFrame
            y: pandas Series
        """
        self.pipeline = None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)
        self.y1_train = self.y_train['tst']
        self.y1_test = self.y_test['tst']
        self.y2_train = self.y_train['waso']
        self.y2_test = self.y_test['waso']
        self.y3_train = self.y_train['se']
        self.y3_test = self.y_test['se']


    def set_pipeline(self):
        """defines the pipeline as a class attribute"""
        #### Scaling numerical features - categorical features and targets removed from this listt
        self.pipeline = Pipeline([
            ('r_forest_classifier', RandomForestClassifier())
        ])

    def run(self):
        print("Running Models")
        self.set_pipeline()
        #self.mlflow_log_param("model", "Linear")
        self.pipeline.fit(self.X_train, self.y1_train)
        self.evaluate(self.X_test, self.y1_test)
        self.save_model_locally('tst')
        self.pipeline.fit(self.X_train, self.y2_train)
        self.evaluate(self.X_test, self.y2_test)
        self.save_model_locally('waso')
        self.pipeline.fit(self.X_train, self.y3_train)
        self.evaluate(self.X_test, self.y3_test)
        self.save_model_locally('se')
        print("Successfully ran all models")

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        metrics_dict = {}
        y_pred = self.pipeline.predict(X_test)
        precision = precision_score(y_pred, y_test)
        metrics_dict['precision_score'] = precision
        recall = recall_score(y_pred, y_test)
        metrics_dict['recall_score'] = recall
        f1 = f1_score(y_pred, y_test)
        metrics_dict['f1_score'] = f1
        accuracy = accuracy_score(y_pred, y_test)
        metrics_dict['accuracy_score'] = accuracy
        print(metrics_dict)
        return metrics_dict

    def save_model_locally(self, model_name):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, f'{model_name}.joblib')
        print(f"{model_name}.joblib saved locally")

if __name__ == "__main__":
    df, data_df = get_data()

    data_df = clean_data(df, data_df, 90, duplicates = True)
    
    data_df, targs = preproc_gen(data_df)

    X = preproc_x(data_df, targs, 0.975)
    y = preproc_y(data_df, 50, 85, 95, 360, 410)

    # Train and save model locally
    trainer = Trainer(X=X, y=y)
    trainer.run()

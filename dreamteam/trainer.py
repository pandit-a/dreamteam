import joblib
from getdata import get_data, clean_data, preproc_gen, preproc_y
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer


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
        "set numerical features"

        numeric_features = [
            'hdl', 'ldl', 'total_cholesterol', 'triglycerides', 'weightkg',
            'hipgirthm', 'neckgirthm', 'waistgirthm', 'waisthip', 'sitsysm',
            'sitdiam', 'packs_week', 'pack_years', 'naps', 'snore_freq',
            'num_pregnancies', 'age', 'heightcm', 'caffeine', 'alcohol_wk',
            'eval_general', 'eval_life', 'eval_health', 'snore_vol', 'choke_freq',
            'apnea_freq', 'awake_freq', 'cups_coffee'
        ]

        "set categorical features as features which aren't numerical or target"

        categorical_features = ['sex', 'race', 'education_survey1',
            'nasal_cong_none', 'any_cvd', 'hypertension_ynd', 'stroke_ynd',
            'asthma_ynd', 'thyroid_ynd', 'thyroid_problem',
            'arthritis_ynd', 'emphysema_ynd', 'menopausal_status',
            'asthma_med', 'cholesterol_med', 'depression_med',
            'htn_med', 'decongestants_med', 'antihistamines_med',
            'anxiety_med', 'diabetes_med',  'sedative_med',
            'thyroid_med'
        ]
        # 'diabetes_ynd',
        # 'apnea',
        #'narcotics_med'
        # 'androgen_med'
        # 'stimulants_med'

        "categorical transformer"

        categorical_transformer = Pipeline(
            steps=[('imputer', SimpleImputer(strategy='most_frequent'))])

        "categorical transformer 2"
        ohe_features = ['hormone_therapy']

        ohe_transformer = Pipeline(
            steps=[('imputer_2', SimpleImputer(strategy='constant', fill_value='Unknown')),
                ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))]
        )

        "numerical transformer"

        numeric_transformer = Pipeline(
            steps=[('imputer',
                    SimpleImputer(strategy='mean')), ('scaler', RobustScaler())])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features),
                ('ohe_transform', ohe_transformer, ohe_features)], remainder='drop')

        prepipe = Pipeline(steps=[('preprocessor', preprocessor)])
        
        """defines the pipeline as a class attribute"""
        #### Scaling numerical features - categorical features and targets removed from this listt
        self.pipeline = Pipeline([
            ('preproc_pipeline', prepipe),
            ('r_forest_classifier', RandomForestClassifier())
        ])

    def run(self):
        print("Running Models")
        self.set_pipeline()
        
        # TST Pipeline
        self.pipeline.fit(self.X_train, self.y1_train)
        self.evaluate(self.X_test, self.y1_test)
        self.save_model_locally('tst_pipeline')
        
        # WASO Pipeline
        self.pipeline.fit(self.X_train, self.y2_train)
        self.evaluate(self.X_test, self.y2_test)
        self.save_model_locally('waso_pipeline')

        #SE PIPELINE
        self.pipeline.fit(self.X_train, self.y3_train)
        self.evaluate(self.X_test, self.y3_test)
        self.save_model_locally('se_pipeline')

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
    #print(df.shape, data_df.shape)

    data_df = clean_data(df, data_df, 90, duplicates = True)
    
    y = preproc_y(data_df, 50, 85, 95, 360, 410)
    
    data_df, targs = preproc_gen(data_df)
    
    X = data_df
    #X = preproc_x(data_df, targs, 0.975)

    # Train and save model locally
    trainer = Trainer(X=X, y=y)
    trainer.run()

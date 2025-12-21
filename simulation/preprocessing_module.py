from sklearn.base import TransformerMixin 
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from functools import reduce


class Eliminator(TransformerMixin):
    def __init__(self, features):
        self.features = features
    def fit(self, X, Y=None):
        return self
    def transform(self, X):
        X_transformed = X.drop(self.features, axis = 1, inplace = False)
        return  X_transformed
    
class Encoder(OrdinalEncoder):
    def __init__(self, to_encode_columns):
        self.to_encode_columns = to_encode_columns
        super().__init__()
    def fit(self, X, Y=None):
        super().fit(X[self.to_encode_columns])
        return self
    def transform(self, X):
        X_transformed = X.copy()
        encoded_data = super().transform(X[self.to_encode_columns])
        X_transformed[self.to_encode_columns] = encoded_data
        return X_transformed    
    

class Operation():
    def __init__(self, function, separator : str):
        self.function = function
        self.separator = "_" + separator + "_"

    def result_col_name(self, *cols):
        return self.separator.join(cols)

class Abstract_synthesis():
    def __init__(self, cols, operation : Operation):
        self.cols = cols
        self.col = operation.result_col_name(*self.cols)
        self.func = operation.function

    def extract_feature(self, X):
        new_feature = reduce(self.func, [X[col] for col in self.cols])
        return new_feature

    def apply_feature(self, X):
        X[self.col] = self.extract_feature(X)

DIV = Operation(lambda x, y: x/y if y != 0 else x, "divided_by")
MUL = Operation(lambda x, y: x*y, "multiplied_by")
CDIV = Operation(lambda x, y: x/(y + 1), "divided_by_plus_one")
SUM = Operation(lambda x, y: x + y, "sum")

class SynthesisReactor(TransformerMixin):
    def __init__(self, params):
        self.params = params

    def fit(self, X, Y=None):
        self.params = [Abstract_synthesis(cols, op) for cols, op in self.params]
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for ob in self.params:
            ob.apply_feature(X_transformed)
        return X_transformed


class CustomSTDScaler(StandardScaler):
    def __init__(self, keep_features):
        self.columns = None
        self.keep_features = keep_features
        super().__init__()
    def fit(self, X, Y=None):
        self.columns = [col for col in X.columns if col not in self.keep_features]
        super().fit(X[self.columns])
        return self
    def transform(self, X):
        X_transformed = X.copy()
        transformed_data = super().transform(X[self.columns])
        X_transformed[self.columns] = transformed_data
        return pd.DataFrame(X_transformed)    


from sklearn.preprocessing import OneHotEncoder

class OneHotEncoderCustom(OneHotEncoder):
    def __init__(self, to_encode_columns):
        self.to_encode_columns = to_encode_columns
        super().__init__()
    def fit(self, X, Y=None):
        super().fit(X[self.to_encode_columns])
        return self
    def transform(self, X):
        X_transformed = X.copy()
        encoded_data = super().transform(X[self.to_encode_columns]).toarray()
        X_transformed = X_transformed.drop(columns=self.to_encode_columns)
        encoded_df = pd.DataFrame(encoded_data, columns=self.get_feature_names_out(self.to_encode_columns), index=X.index)  # Add index
        X_transformed = pd.concat([X_transformed, encoded_df], axis=1)
        return X_transformed




test_num = 57


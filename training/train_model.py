import os
import sys
import django
import pandas as pd
import joblib
import json
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# プロジェクトのルートディレクトリをPythonのパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Djangoのセットアップ
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

# train_model.py専用の、学習機能を持つEvaluaterクラスを定義
class Evaluater():
    drops = ["馬番"]
    X = None
    y = None
    X_train = None
    X_val = None
    X_test = None
    y_train = None
    y_test = None
    y_val = None
    params = {'objective': 'regression',
                     'random_state': 57,
                     'metric': 'l2',
                     'feature_pre_filter': False,
                     'lambda_l1': 0.15883047646498394,
                     'lambda_l2': 9.85103023641964,
                     'num_leaves': 4,
                     'feature_fraction': 0.5,
                     'bagging_fraction': 0.9223910437388337,
                     'bagging_freq': 5,
                     'min_child_samples': 20,
                     'num_iterations': 1000,
                     'verbose': -1}
    model = lgb.LGBMRegressor(**params)
    
    
    def __init__(self, X, y, odds, test_size=0.3):
        race = list(X.index.unique())
        race_tmp, race_test = train_test_split(race, shuffle=True, test_size=test_size)
        race_train, race_val = train_test_split(race_tmp, shuffle=True, test_size=0.3)
        self.X_train, self.y_train = X[X.index.isin(race_train)], y[X.index.isin(race_train)]
        self.X_test, self.y_test = X[X.index.isin(race_test)], y[X.index.isin(race_test)]
        self.X_val, self.y_val = X[X.index.isin(race_val)], y[X.index.isin(race_val)]
        self.X = X
        self.y = y
        if odds:
            self.drops.append("単勝")
        return
    
    def fit(self,):
        verbose_eval = -1
        self.model.fit(
            self.X_train.drop(self.drops, axis=1, inplace=False), self.y_train, 
            eval_metric='mean_squared_error', 
            eval_set=[(self.X_val.drop(self.drops, axis=1, inplace=False), self.y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=10, 
                        verbose=False),
                    lgb.log_evaluation(verbose_eval)]
                 )

def train_and_save_model():
    """
    モデルを学習し、ファイルに保存する関数
    """
    print("モデルの学習を開始します...")

    # data/data.pickleからデータをロード
    data_path = "data/data.json"
    data = pd.read_json(data_path, orient='records')
    # race_idカラムをインデックスに設定
    data = data.set_index('race_id')
    data.index = pd.to_datetime(data.index, format='%Y%m%d%H%M') # 時刻型に変換

    # ロードされた'data'を元に、特徴量とターゲットを準備
    X_full, y_full = data.drop(["着順"], axis=1), -data["着順"]

    # このスクリプト内のEvaluaterクラスをインスタンス化
    evaluator = Evaluater(X_full, y_full, odds=False)

    # モデルを学習
    evaluator.fit() 

    # --- モデルとカラムリストの保存 ---
    model_path = "data/models/lgbm_model.joblib"
    columns_path = "data/models/model_columns.json"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # モデルをjoblib形式で保存
    joblib.dump(evaluator.model, model_path)
    print(f"モデルの保存が完了しました。モデルは {model_path} に保存されました。")

    # カラムリストをJSON形式で保存
    model_columns = list(evaluator.X_train.drop(evaluator.drops, axis=1).columns)
    with open(columns_path, 'w') as f:
        json.dump(model_columns, f)
    print(f"カラムリストの保存が完了しました。リストは {columns_path} に保存されました。")


if __name__ == '__main__':
    train_and_save_model()
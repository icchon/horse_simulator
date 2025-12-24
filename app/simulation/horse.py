import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_squared_error
from sklearn.model_selection import KFold

import simulation.preprocessing_module as pm
from simulation.forms import MyForm




class Cal:
    @staticmethod
    def cal_tansho(x):
        race_id = x.name
        


        horse_number = int(x["馬番"])
        odds = x[1]
        df = pay_dict[race_id]
        df_tansho = df.loc["単勝"].copy() 
        df_tansho["該当馬"] = list(map(int, df_tansho["該当馬"]))
        df_tansho["金"] = list(map(str, df_tansho["金"]))
        invested = 100 
        payback = 0
        if horse_number in df_tansho["該当馬"]:
            idx = df_tansho["該当馬"].index(horse_number)
            tmp = df_tansho["金"][idx]
            tmp = tmp.split(",")
            payback += int("".join(tmp))
        return (payback, invested)

    @staticmethod    
    def cal_fukusho(x):
        race_id = x.name
        pay_df = pay_dict[race_id]
        df_fukusho = pay_df.loc["複勝"].copy()
        df_fukusho["該当馬"] = list(map(int, df_fukusho["該当馬"]))
        df_fukusho["金"] = list(map(str, df_fukusho["金"]))
        horse_number = int(x["馬番"])
        invested = 100
        payback_sum = 0
        if horse_number in df_fukusho["該当馬"]:
            idx = df_fukusho["該当馬"].index(horse_number)
            tmp = df_fukusho["金"][idx]
            tmp = tmp.split(",")
            tmp = int("".join(tmp))
            payback_sum += tmp
        return (payback_sum, invested)

class Evaluater():
    """
    学習済みのモデルを使い、予測と評価を行うクラス。
    """
    def __init__(self, model, X_test_for_predict, X_test_for_cal):
        self.model = model
        self.X_test_predict = X_test_for_predict
        self.X_test_cal = X_test_for_cal       

    def predict(self, threshold=0):
        """学習済みモデルで予測を行う"""
        pred = self.model.predict(self.X_test_predict)
        df = pd.DataFrame(pred, index=self.X_test_predict.index, columns=["pred"])
        df["mean"] = df.groupby(df.index)["pred"].transform("mean")
        df["std"] = df.groupby(df.index)["pred"].transform("std")
        df["pred_scaled"] = (df["pred"] - df["mean"]) / df["std"]
        
        final_pred = np.zeros(len(df))
        final_pred[df["pred_scaled"] >= threshold] = 1
        final_pred[df["pred_scaled"] < threshold] = 0
        return final_pred

    def cal(self, threshold=0, fukusho=False, tansho=False):
        """予測結果に基づいて収益を計算する"""
        if not tansho and not fukusho:
            raise ValueError("賭け方を指定してください") 
        if sum([tansho, fukusho]) != 1:
            raise ValueError("賭け方の指定は一つだけです")
        
        pred = self.predict(threshold)
        purchased_mask = (pred == 1)
        purchased = self.X_test_cal[purchased_mask]
        
        invested = 0
        payback_sum = 0

        if not purchased.empty:
            if tansho:
                results = purchased.apply(lambda x: Cal.cal_tansho(x), axis=1)
                paybacks = results.str[0]
                investeds = results.str[1]
                invested += investeds.sum()
                payback_sum += paybacks.sum()
            elif fukusho:
                results = purchased.apply(lambda x: Cal.cal_fukusho(x), axis=1)
                paybacks = results.str[0]
                investeds = results.str[1]
                invested += investeds.sum()
                payback_sum += paybacks.sum()
                    
        res = {
            "invested": invested,
            "payback_sum": payback_sum,
            "div": payback_sum - invested,
            "kaishuuritu": (payback_sum / invested) * 100 if invested > 0 else 100
        }
        return res
    
    def visualize(self, bins=20, fukusho=False, tansho=False):
        """異なる閾値でシミュレーションを実行し、可視化用のデータを生成する"""
        res = {"threshold": [], "kaishuuritu": [], "div": [], "invested": []}
        for i in reversed(range(bins + 2)):
            threshold = -3 + (i / bins) * 6
            info = self.cal(threshold=threshold, tansho=tansho, fukusho=fukusho)
            res["threshold"].append(threshold)
            res["div"].append(info["div"])
            res["invested"].append(info["invested"])
            res["kaishuuritu"].append(info["kaishuuritu"])
        
        for key in res:
            res[key] = np.array(res[key])

        total_invested = np.sum(res["invested"])
        if total_invested > 0:
            res["bet_percent"] = res["invested"] * 100 / total_invested
            for i in range(len(res["bet_percent"]) - 1):
                res["bet_percent"][i + 1] += res["bet_percent"][i]
        else:
            res["bet_percent"] = np.zeros(len(res["invested"]))
            
        return res

def get_simulation_dict(n_bins=20, features={}, test_size=0.3, feature_combinations=[]):
    """
    シミュレーションを実行し、結果を返すメイン関数。
    データ分割、モデル学習、評価を行う。
    """
    X, y = data.drop(["着順"], axis=1), -data["着順"]
    
    available_race_ids = [pd.to_datetime(rid, format='%Y%m%d%H%M') for rid in pay_dict.keys()]

    X_filtered = X[X.index.isin(available_race_ids)]
    y_filtered = y[y.index.isin(available_race_ids)]

    race_ids = X_filtered.index.unique()
    
    if len(race_ids) == 0:
        print("Warning: No common race_ids found between data.json and pay_dict.json. Returning empty simulation results.")
        return {"threshold": [], "kaishuuritu": [], "div": [], "invested": [], "bet_percent": []}, []

    train_ids, test_ids = train_test_split(race_ids, test_size=test_size, random_state=42)
    
    X_train_raw, X_test_raw = X_filtered[X_filtered.index.isin(train_ids)], X_filtered[X_filtered.index.isin(test_ids)]
    y_train, y_test = y_filtered[y_filtered.index.isin(train_ids)], y_filtered[y_filtered.index.isin(test_ids)]

    # ユーザーが選択した特徴量を決定
    use_features = ["馬番", "単勝", "セ"] 
    for feature, is_used in features.items():
        if is_used:
            add_feature = MyForm.param_to_feature.get(feature)
            if add_feature and add_feature not in use_features:
                use_features.append(add_feature)

    # 特徴量合成で使われるカラムを use_features に追加
    if len(feature_combinations) > 0:
        # feature_combinations は [(['feature1', 'feature2'], '*'), ...] という形式
        for features_list, op_str in feature_combinations:
            for feature_name in features_list:
                if feature_name not in use_features:
                    use_features.append(feature_name)

    # 学習用とテスト用のデータフレームを、選択された特徴量でフィルタリング
    X_train_filtered = X_train_raw[use_features]
    X_test_filtered = X_test_raw[use_features]

    # 特徴量合成を適用
    if len(feature_combinations) > 0:
        op_map = {'*': pm.MUL, '+': pm.SUM, '/': pm.DIV}
        synthesis_features = []
        # feature_combinations は [(['feature1', 'feature2'], '*'), ...] という形式
        for features_list, op_str in feature_combinations:
            op = op_map.get(op_str, pm.MUL) # マップにない場合はデフォルトで掛け算
            synthesis_features.append((features_list, op))
        
        sr = pm.SynthesisReactor(synthesis_features)
        X_train_filtered = sr.fit_transform(X_train_filtered)
        X_test_filtered = sr.transform(X_test_filtered) # テストデータにはtransformのみ適用

    # モデル学習
    params = {
        'objective': 'regression',
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
        'verbose': -1
    }
    model = LGBMRegressor(**params)

    # 学習に不要なカラムを除外
    features_for_training = [f for f in X_train_filtered.columns if f not in ["馬番"]]
    
    model.fit(X_train_filtered[features_for_training], y_train)

    print(f"DEBUG: Features for training: {features_for_training}")

    # 評価
    # Evaluaterに渡すX_testを、学習済みモデルの列に合わせる
    X_test_for_predict = X_test_filtered[features_for_training]
    
    # X_test_for_calは元のフィルタリングされたデータフレームを渡す(馬番など計算に必要な情報を含むため)
    ev = Evaluater(model, X_test_for_predict, X_test_filtered)
    
    output = ev.visualize(tansho=True, bins=n_bins)
    return output, features_for_training




from django import forms






features = ['着順', '馬番', '斤量', '単勝', '人気', 'length', '騎乗回数', '勝率', '連対率', '複勝率',
       '賞金_ave', '賞金_sum', '順番_ave', '賞金_ave_2', '賞金_sum_2', '順番_ave_2',
       'length_match', 'race_type_match', 'horse_ped_score', 'タイム/length',
       'score', '年齢', '体重', '増減', 'セ', '牝', '牡', '勝率*騎乗回数', '連対率*騎乗回数',
       '複勝率*騎乗回数', '増減/体重', '斤量/体重']

class MyForm(forms.Form):
    param_to_feature = {
        "parameter1": "length",
        "parameter2": "斤量",
        "parameter3": "単勝",
        "parameter4": "体重",
        "parameter5": "人気",
        "parameter6": "勝率",
        "parameter7": "騎乗回数",
        "parameter8": "年齢",
        "parameter9": "賞金_ave",
        "parameter10": "賞金_sum",
        "parameter11": "順番_ave",
        "parameter12": "増減",
        "parameter13": "horse_ped_score",
    }

    feature_to_display = {
        "length": "距離",
        "斤量": "斤量",
        "単勝": "オッズ",
        "体重": "馬の体重",
        "人気": "人気",
        "勝率": "勝率",
        "騎乗回数": "騎乗回数",
        "年齢": "年齢",
        "賞金_ave": "平均賞金額",
        "賞金_sum": "総賞金額",
        "順番_ave": "平均順位",
        "増減": "体重の増減",
        "horse_ped_score": "遺伝子",
    }

    display_to_feature = {}
    for feature, display in feature_to_display.items():
       display_to_feature[display] = feature

    display_to_help_text = {
       "距離":"レースの距離を示すパラメータです",
       "斤量":"斤量を示すパラメータです",
       "オッズ":"馬の単勝オッズを示すパラメータです",
       "馬の体重":"馬の体重を示すパラメータです",
       "人気":"〇番人気を示すパラメータです",
       "勝率":"直近一年の勝率を示すパラメータです",
       "騎乗回数":"ジョッキーの総騎乗回数を示すパラメータです",
       "年齢":"馬の年齢を示すパラメータです",
       "平均賞金額":"馬の直近一年の平均賞金額を示すパラメータです",
       "総賞金額":"馬の直近一年の総賞金額を示すパラメータです",
       "平均順位":"馬の直近一年の平均順位を示すパラメータです",
       "体重の増減":"レース時に馬がいつもの体重からどれだけ増減しているかを示すパラメータです",
       "遺伝子":"馬の遺伝子の強さを示すパラメータです",
    }


    param_to_feature_display = {}
    for key, value in param_to_feature.items():
       param_to_feature_display[key] = feature_to_display[value]
    param_to_help_text = {}    
    for key, value in param_to_feature_display.items():
       param_to_help_text[key] = display_to_help_text[value]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.param_to_feature_display.items():
            self.fields[key] = forms.BooleanField(required=False, label=value, help_text=self.param_to_help_text[key], initial=True)
    def __len__(self):
       return len(self.fields)        

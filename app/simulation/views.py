from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from simulation.horse import get_simulation_dict
import json
import numpy as np
import sys
import os
from .forms import MyForm
import re


def home(request):
   return render(request, "simulation/home.html")

speed_to_bins = {
   "low":40, # グラフの粒度を細かく
   "medium":20,
   "high":10, # グラフの粒度を粗く
}

speed_to_test_size = {
    "low":0.3, # 予測データ量を多く
    "medium":0.3,
    "high":0.1, # 予測データ量を少なく
}

def heavy_function(features, speed, feature_combinations):
    n_bins = speed_to_bins[speed]
    test_size = speed_to_test_size[speed]
    
    output, columns = get_simulation_dict(n_bins=n_bins, features=features, test_size=test_size, feature_combinations=feature_combinations)
    labels = list(output["bet_percent"])
    data = list(output["kaishuuritu"])
    
    
    #print(columns)
    return labels, data

def parse_generated_feature(feature_string):
    # '・' を削除し、前後の空白を除去
    s = feature_string.replace('・', '').strip()
    # ' * ', ' + ', ' / ' のいずれかで文字列を分割する。演算子もキャプチャする。
    parts = re.split(r' ([\*+\/]) ', s)
    
    # 期待される形式: ['feature1', '*', 'feature2']
    if len(parts) < 3:
        return None, None
    
    # 偶数インデックスが特徴量名、奇数インデックスが演算子
    features = [parts[i] for i in range(0, len(parts), 2)]
    operator = parts[1]
    
    return features, operator


def simulation(request):
    if request.method == "POST":
        print(f"DEBUG: Request Body: {request.body}") # デバッグプリント
        # JSONデータを解析
        try:
            json_data = json.loads(request.body)
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON"}, status=400)

        form = MyForm(data=json_data) # JSONデータをフォームにバインド

        if form.is_valid():
            features = {}
            for field_name, field_value in form.cleaned_data.items():
                if field_value: # BooleanFieldなのでTrue/Falseで判定
                    features[field_name] = field_value

            # generated_features はJSONデータから直接取得
            generated_features_str = json_data.get('generated_features', '')
            generated_features = generated_features_str.split(',')
            feature_combinations = []
            for generated_feature in generated_features:
                if not generated_feature:
                    continue
                
                display_names, operator = parse_generated_feature(generated_feature)
                
                if display_names and operator:
                    # 表示名から内部名に変換
                    internal_names = [MyForm.display_to_feature.get(display) for display in display_names]
                    
                    # 全ての特徴量名が正しく変換できたかチェック (Noneが含まれていないか)
                    if all(name is not None for name in internal_names):
                        feature_combinations.append((internal_names, operator))
                    
            labels = []
            data = []
            speed = json_data.get("speed") # JSONデータからspeedを取得
            if speed not in speed_to_bins:
                speed = "low"  # デフォルト値
            labels, data = heavy_function(features=features, speed=speed, feature_combinations=feature_combinations)
            
            my_dict = {
                "labels": labels,
                "data": data,
                "form": form,
            }

            print("--------------------------------------")

            return render(request, "simulation/simulation.html", my_dict)
    else:
        initial_data = {key: True for key in MyForm.param_to_feature.keys()}
        form = MyForm(initial=initial_data)
        my_dict = {
            "labels": [],
            "data": [],
            "form": form,
        }
        return render(request, "simulation/simulation.html", my_dict)
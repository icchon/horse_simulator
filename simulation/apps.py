import pandas as pd
import joblib
import json
import os
from django.apps import AppConfig
from django.conf import settings

# 読み込むモジュールをここでインポート
import simulation.horse

class SimulationConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'simulation'

    def ready(self):
        # 循環参照を避けるため、ready()内でインポート
        
        try:
            simulation.horse.PRETRAINED_MODEL = joblib.load(MODEL_PATH)
            # print(f"Model loaded from {MODEL_PATH}")

            # カラムリストをロードし、simulation.horseモジュールに設定
            with open(COLUMNS_PATH, 'r') as f:
                simulation.horse.MODEL_COLUMNS = json.load(f)
            # print(f"Model columns loaded from {COLUMNS_PATH}")
        except FileNotFoundError:
            simulation.horse.PRETRAINED_MODEL = None
            simulation.horse.MODEL_COLUMNS = None
            # print(f"Warning: Model or column file not found. Please ensure {MODEL_PATH} and {COLUMNS_PATH} exist.")
        except Exception as e:
            # print(f"Error loading model or columns: {e}")
            simulation.horse.PRETRAINED_MODEL = None
            simulation.horse.MODEL_COLUMNS = None

        # --- data.json のロード ---
        DATA_JSON_PATH = os.path.join(settings.BASE_DIR, "data", "data.json")
        try:
            data = pd.read_json(DATA_JSON_PATH, orient='records')
            data = data.set_index('race_id')
            data.index = pd.to_datetime(data.index, format='%Y%m%d%H%M')
            simulation.horse.data = data
            # print(f"Data loaded from {DATA_JSON_PATH}")
        except FileNotFoundError:
            simulation.horse.data = None
            # print(f"Warning: data.json not found at {DATA_JSON_PATH}")
        except Exception as e:
            # print(f"Error loading data.json: {e}")
            simulation.horse.data = None

        # --- pay_dict.json のロード ---
        PAY_DICT_JSON_PATH = os.path.join(settings.BASE_DIR, "data", "pay_dict.json")
        try:
            with open(PAY_DICT_JSON_PATH, 'r', encoding='utf-8') as f:
                pay_dict_json_data = json.load(f)
            
            pay_dict = {}
            for race_id_str, df_json_records in pay_dict_json_data.items():
                df = pd.read_json(json.dumps(df_json_records), orient='records')
                df = df.set_index('index') # 'index'カラムをインデックスとして設定
                pay_dict[pd.to_datetime(race_id_str)] = df
            simulation.horse.pay_dict = pay_dict
            # print(f"Pay dictionary loaded from {PAY_DICT_JSON_PATH}")
        except FileNotFoundError:
            simulation.horse.pay_dict = {}
            # print(f"Warning: pay_dict.json not found at {PAY_DICT_JSON_PATH}")
        except Exception as e:
            # print(f"Error loading pay_dict.json: {e}")
            simulation.horse.pay_dict = {}

        # print("--- Simulation AppConfig ready() finished ---")

<p align="center"><h1 align="center">Horse Simulator</h1></p>

<p align="center">DjangoとLightGBMを用いた競馬予測シミュレーションWebアプリケーション</p>

<p align="center"><a href="./LICENSE"><img src="https://img.shields.io/github/license/icchon/horse_sumilator" alt="license"></a></p>

<br>

## Contents

- [Overview](#overview)

- [Features](#features)

- [Technology Stack](#technology-stack)

- [Getting Started](#getting-started)

- [License](#license)

---

## Overview

`Horse Simulator`は、DjangoとLightGBMを用いて構築された競馬予測シミュレーションWebアプリケーションである。ユーザーが選択した特徴量に基づき、過去のレースデータを用いて機械学習モデルのパフォーマンス（回収率など）をシミュレーションする。

---

## Features

- **Web UI:** 直感的なWebインターフェースでシミュレーションのパラメータを設定可能である。

- **特徴量選択:** 複数の競馬関連特徴量（斤量、人気、騎手勝率など）から、モデルに投入するものを自由に選択できる。

- **特徴量生成:** 選択した特徴量を組み合わせて、新しい特徴量を動的に生成する機能を持つ。

- **機械学習モデル:** `LightGBM`（勾配ブースティング）による回帰モデルを実装している。

- **結果の可視化:** `Chart.js`を使用し、賭けた割合に対する回収率の変化を折れ線グラフで表示する。

---

## Technology Stack

- **Backend:** Django

- **Frontend:** HTML, CSS, JavaScript, Chart.js

- **ML/Data Science:** Scikit-learn, Pandas, LightGBM

---

## Getting Started

### Prerequisites

- Python 3.x

- `requirements.txt`に記載されたライブラリ

### Installation

```sh

❯ git clone https://github.com/icchon/horse_sumilator

❯ cd horse_sumilator

❯ pip install -r requirements.txt

```

### Usage

```sh

❯ python manage.py runserver

```

開発サーバーを起動し、Webブラウザで指定されたURL（例: `http://127.0.0.1:8000/`）にアクセスする。

---

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

© 2025 icchon

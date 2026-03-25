<h1 align="center">🎯 YOLOリアルタイム顔認識システム</h1>

<p align="center">
  <strong>YOLOv11 + MobileFaceNet による本格的なリアルタイム顔認識デスクトップアプリ</strong><br/>
  Windows · macOS · Linux · Raspberry Pi 対応
</p>

<p align="center">
  <a href="../README.md">🇬🇧 English</a> ·
  <a href="README_zh-TW.md">🇹🇼 繁體中文</a> ·
  <a href="README_zh-CN.md">🇨🇳 简体中文</a> ·
  <a href="README_ja.md">🇯🇵 日本語</a>
</p>

---

## ⬇️ ダウンロード

**[Releases](https://github.com/Wang200935/YOLO-Real-Time-Face-Recognition/releases)** ページから最新版をダウンロードしてください：

| プラットフォーム | ファイル |
|---|---|
| 🪟 Windows | `YOLO-FaceRecognition.exe` |
| 🍎 macOS | `YOLO-FaceRecognition.dmg` |
| 🐧 Linux / 開発者 | 下記の[ソースから実行](#ソースから実行)を参照 |

> モデルファイルは**初回起動時に自動ダウンロード**されます（約 22 MB）。

---

## ✨ 主な機能

| | |
|---|---|
| 🎯 **即時認識** | 登録済みの顔が見えた瞬間に認識 |
| 🧠 **自己学習** | カメラ運用中に新しい角度を自動収集 |
| 🔄 **360° カバー** | 正面・横顔・俯仰角を自動収集し、頭を向けても認識可能 |
| 📸 **スマートキャプチャ** | 鮮明なフレームのみ保存、ブレたフレームは自動廃棄 |
| 🕴️ **全身シーン対応** | 全身が映っていても顔を正確に認識 |
| 🚫 **誤登録防止** | 横顔が別人として重複登録されるのを防止 |
| 🔔 **アラート機能** | 特定の人物を検出した際に通知 |
| 🖥️ **デスクトップGUI** | モデル切替・閾値調整・人物管理に対応 |

---

## 🚀 ソースから実行

### 要件

- Python **3.10+**
- USB または内蔵カメラ
- RAM 4 GB 推奨

### セットアップ

```bash
git clone https://github.com/Wang200935/YOLO-Real-Time-Face-Recognition.git
cd YOLO-Real-Time-Face-Recognition
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

---

## 🎓 人物の登録方法

### 方法 A — ライブカメラ
1. **▶ 認識開始** をクリック
2. カメラの前に約 3 秒立つ
3. 自動でビジターとして登録され、角度を学習し続ける

### 方法 B — 写真のアップロード（推奨）
1. **➕ 人物追加** → 名前を入力 → 写真を選択
2. 以下の角度で **8〜15 枚** 用意することを推奨：

| 角度 | ヒント |
|---|---|
| 正面 | カメラを直視 |
| 左右 45° | 頭を少し横に向ける |
| 左右 90° | 完全に横を向く |
| 上/下向き | 頭を少し上下に傾ける |
| 全身 | 2〜3 メートル離れた全身 |

---

## ⚙️ 設定

アプリの**設定**画面から調整できます：

| 項目 | デフォルト | 効果 |
|---|---|---|
| 認識閾値 | 0.64 | 低いほど厳格 |
| スキップフレーム | 2 | 高いほど省エネ |
| 検出信頼度 | 0.45 | 低いほど遠距離も検出 |
| 検出解像度 | 640 | 低性能機は 320 推奨 |
| GPU 加速 | オフ | `cuda:0` で有効化 |

---

## ❓ よくある質問

**カメラが開かない** → 他のアプリがカメラを使用中の可能性。カメラインデックスを `1` に変更してみてください。macOS は System Settings → Privacy → Camera で許可が必要です

**認識精度が低い** → より多くの角度の写真で再登録（8〜15 枚推奨）。認識閾値を下げる

**FPS が低い** → 検出解像度を 320 に下げ、スキップフレームを 4 に増やす

**モデルのダウンロードが失敗する** → インターネット接続を確認。またはモデルファイルを手動で `data/models/` に配置

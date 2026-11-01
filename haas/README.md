# Haas-ja-v1.0 を Hex で使う（自分用フォーク）

[Hex](https://github.com/anomalyco/hex) の音声認識に
[Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0)（Third Intelligence, Apache-2.0）を使うための、
[anomalyco/transcribe.cpp](https://github.com/anomalyco/transcribe.cpp) の自分用フォークです。本家への PR は出しません。

## 構成

- ブランチ `haas` = Hex が固定している transcribe.cpp のコミット（`haas/BASE` に記録）+ Haas 対応のコミット。
  - `src/arch/parakeet/kda_predictor.{h,cpp}`（新規）: KDA（Kimi Delta Attention）予測ネット。1 トークンずつ CPU の ggml グラフで実行（`ggml_gated_delta_net` の KDA モード）。
  - 既存ファイルの変更は呼び出し口だけ（`decoder.{h,cpp}` `weights.{h,cpp}` `src/CMakeLists.txt` `tools/transcribe-quantize/policy.cpp`）。KDA の状態は `LstmState::kda` に入るので、TDT / RNN-T / ストリーミングのデコーダ、語彙リスコアリングは無変更で動く。
  - `scripts/convert-parakeet.py`: `Haas-ja-v1.0` プロファイル。`.nemo` を NeMo なしで直接読み、`pred.kda.*` と `stt.parakeet.predictor.kind = "kda"` を書く。
  - `haas/`: このフォーク専用のツール（下記）。
- Hex 本体はフォークしません。`haas/hex/update.sh` が本家 Hex を `~/haas-hex/hex` に取得し、
  `Cargo.toml` の `[patch.crates-io]` をこのリポジトリの作業ツリー（`~/haas-hex/transcribe.cpp-build`）に向ける
  ローカルコミットを 1 つ載せてビルドします（パスが入るので GitHub には置かない前提）。

| ファイル | 役割 |
|---|---|
| `haas/convert.sh` | HF から `.nemo` を取得 → F32 GGUF → Q8_0 GGUF（Mac / Linux、NeMo・GPU 不要） |
| `haas/parity.sh` | 実重みで C++ と PyTorch 参照を数値比較（Linux のみ: fla が Triton を要求） |
| `haas/hex/create-signing-identity.sh` | 自己署名のコード署名証明書「Hex Local」を作る（初回のみ） |
| `haas/hex/update.sh` | Hex のリリースに追従してビルド・テスト・署名・インストール（下記） |
| `haas/kda_reference.py` | 参照実装: Haas-ASR の `prednet.py` + fla 0.5.2 の本物のモジュール。GPU 用の Triton カーネル 3 つだけ fla 自身の torch 実装に差し替え |
| `haas/make_synthetic_nemo.py` | 本物と同じ予測ネット形状・パラメータ名のランダム重み `.nemo`（実重みなしでのテスト用） |
| `haas/kda_parity.cpp`, `haas/build_parity.sh`, `haas/check_parity.py` | 数値比較のハーネス |

## 検証状況

- 合成モデル（予測ネットは本物と同じ形状、重みはランダム）で、C++ と参照実装が一致:
  64 ステップの予測ネット出力の相対誤差 1.3e-6（F32）、TDT greedy デコード（200 フレーム）の 129 トークンが完全一致。
  状態の持ち越しを壊す変異を入れると失敗することも確認済み。
- transcribe.cpp 既存のテスト 62 件はすべて通過（LSTM の Parakeet は従来どおり）。
- Q8_0 では予測ネット出力に rms 約 2% の差（線形層の量子化。Parakeet の LSTM と同じ方針）。
- **実重みでの検証はまだ**（作業したクラウド環境から Hugging Face に接続できなかったため）。下の手順 2〜3 で行う。

## 手順（Mac）

前提: Xcode（コマンドラインツール含む）、Rust（rustup）、CMake、jq、[uv](https://docs.astral.sh/uv/)、git。

```bash
git clone <このフォーク> ~/haas-hex/transcribe.cpp && cd ~/haas-hex/transcribe.cpp
git checkout haas
```

### 1. 署名用の証明書を作る（初回のみ）

```bash
haas/hex/create-signing-identity.sh
```

ログインパスワードを 1 回聞かれます（コード署名用に信頼する設定）。最初の `codesign` で鍵の使用を聞かれたら「常に許可」。

### 2. モデルを変換する

```bash
haas/convert.sh
```

`~/haas-hex/models/Haas-ja-v1.0-Q8_0.gguf`（Hex 用）と `~/haas-hex/models/Haas-ja-v1.0/Haas-ja-v1.0-F32.gguf`（参照用）ができます。
変換後、すぐに聞いて確かめられます:

```bash
build/bin/transcribe-cli -m ~/haas-hex/models/Haas-ja-v1.0-Q8_0.gguf samples/ja.wav
build/bin/transcribe-cli -m ~/haas-hex/models/Haas-ja-v1.0-Q8_0.gguf samples/fleurs-ja.wav
```

### 3. 実重みで数値検証（Linux VM、推奨）

```bash
git clone https://github.com/third-intelligence/Haas-ASR ../Haas-ASR
haas/convert.sh          # Linux VM 側でも変換（または F32 GGUF をコピー）
haas/parity.sh           # F32: PASS が 2 つ出れば OK
haas/parity.sh third-intelligence/Haas-ja-v1.0 ~/haas-hex/models/Haas-ja-v1.0-Q8_0.gguf   # Q8_0 の差の確認
```

### 4. Hex をビルドしてインストール

```bash
haas/hex/update.sh --ref dev
```

- `--ref` を省くと最新リリース（`vX.Y.Z` タグ）。ただし **2026-11-01 時点の最新リリース v3.1.1 には
  「Labs › Use your own speech model」（anomalyco/hex#759）がまだ入っていない**ので、それを含むリリースが出るまでは `--ref dev`。
- 中身: transcribe.cpp のビルドとテスト（初回は `samples/ja*.wav` の書き起こしを `~/haas-hex/golden/` に保存、以後はそれと比較）
  → Hex のビルド（`HEX_RELEASE_UNSIGNED=1 HEX_LOCAL_BUILD=1 scripts/build-app.sh`: 名前 Hex / bundle `anomaly.hex` / 更新チェックなし）
  → 「Hex Local」で署名し直し（`HexMicrophone.driver` → `scrawl-helper` → アプリ本体、公式の `sign-macos.sh` と同じ順序・entitlements）
  → `/Applications/Hex.app` を置き換えて起動。
- 初回だけ macOS がマイク・アクセシビリティ等の許可を改めて求めます（公式版と署名が違うため）。以後は同じ証明書で署名する限り維持されます。
- `/Applications` のアプリを置き換えるため、ターミナルに「App の管理」の許可を求められることがあります。

### 5. Hex で Haas を選ぶ

Settings → Labs → **Use your own speech model** を ON → モデル選択の **Add model file…** → `~/haas-hex/models/Haas-ja-v1.0-Q8_0.gguf`。

## 更新（Hex の新しいリリースに追従）

```bash
haas/hex/update.sh            # 最新リリース
haas/hex/update.sh --push     # 成功したら rebase 後のブランチを origin の haas に push
```

1. 最新の `vX.Y.Z` タグと、そのリリースが固定している transcribe.cpp のコミットを読む。
2. それが `haas/BASE` と違えば、Haas のコミットを新しいコミットに rebase（新ブランチ `haas-<7桁>`、今のチェックアウトはそのまま）。
   衝突したら `~/haas-hex/rebase-<7桁>` で止まるので、そこで解決（Claude Code に任せてよい）して再実行。
3. transcribe.cpp のビルド・テスト・書き起こし比較 → Hex のビルド・署名・インストール。

書き起こしが変わったと言われたら、聞いて確かめ、問題なければ表示されたファイルを消して再実行（新しい書き起こしが保存される）。

## Windows（GPU なしのノート）

Git Bash から `haas/hex/update.sh --ref dev`（`haas/hex/build-windows.sh` 経由で Hex の `scripts/build-windows.sh` を
`HEX_LOCAL_BUILD=1` で実行し、`dist/hex-*-windows-x64.exe` を作る）。**まだ Windows で試していません。**
Hex の Windows ビルドに必要なもの（Rust MSVC、Visual Studio Build Tools、CMake、jq、x86_64 なら Vulkan SDK）が要ります。
予測ネットはどの環境でも CPU で動きます。

## メモ

- 予測ネットの重みは Q8_0 でもロード時に fp32 に展開されます（`A_log` `dt_bias` ノルム・畳み込みは GGUF 上も F32）。
  CER が ONNX 版（FLEURS ja 4.34%）より明らかに悪いときは、まず F32 GGUF と比べてください。
- Hex の語彙（vocabulary）は、TDT モデルでは書き起こし後のリスコアリングで、KDA でもそのまま動きます（合成モデルで動作確認のみ）。
- 自己署名だと画面収録用の仮想マイク（`HexMicrophone.driver`）は読み込まれない可能性があります（音声入力には不要）。
- 公式 Hex に戻すには、公式の DMG から入れ直します。

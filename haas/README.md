# Haas-ja-v1.0 を Hex で使う（自分用フォーク）

[Hex](https://github.com/anomalyco/hex) の音声認識に
[Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0)（Third Intelligence, Apache-2.0）を使うための、
[anomalyco/transcribe.cpp](https://github.com/anomalyco/transcribe.cpp) の自分用フォークです。本家への PR は出しません。

> 個人用の非公式フォークです。transcribe.cpp・Hex・Haas の各開発元（handy-computer、anomalyco、Third Intelligence）とは関係がなく、
> サポートもありません。モデルの重みは含みません。

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
| `haas/convert.sh` | HF から `.nemo` を取得 → F32 GGUF → Q8_0 GGUF（Mac / Linux / Windows の Git Bash、NeMo・GPU 不要） |
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
- 実重み（bf16 の `.nemo`）で変換・書き起こしを確認済み: `samples/ja.wav` →
  「うちの中学は弁当制で、持って行けない場合は、五十円の学校販売のパンを買う。」（M3 / Metal で実時間の約 40〜50 倍）。
  句読点は付く。実重みでの数値比較（手順 3、Linux）は任意。
- Windows 11（x64、NVIDIA GPU）でも、Hex v3.3.0 に取り込んで日本語の音声入力ができることを確認済み（2026-11-04）。
  `transcribe-cli`（CPU）での `samples/ja.wav` は Mac と同じ書き起こしで、実時間の約 12 倍。

## 手順（Mac）

前提: Xcode（コマンドラインツール含む）、Rust（rustup）、CMake、jq、[uv](https://docs.astral.sh/uv/)、git。
足りないものはスクリプトが最初に教えてくれます。Homebrew があれば CMake と jq は `brew install cmake jq`、
Rust は `curl https://sh.rustup.rs -sSf | sh`。

```bash
git clone https://github.com/raihalea-lab/transcribe.cpp-haas ~/haas-hex/transcribe.cpp && cd ~/haas-hex/transcribe.cpp
```

置き場所は自由です（`ghq get raihalea-lab/transcribe.cpp-haas` でも可）。スクリプトはリポジトリの場所によらず、
作業用の置き場 `~/haas-hex`（環境変数 `HAAS_HOME` で変更可）にモデルや Hex のビルドを作ります。

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

### 3. 実重みで数値検証（Linux VM、任意）

Mac では動きません（参照実装の fla が Triton を必要とし、Triton は macOS 版がない）。Mac では手順 2 の `transcribe-cli` で聞いて確かめれば十分です。
Haas-ASR を ghq で置いた場合は `HAAS_ASR_SRC="$(ghq root)/github.com/third-intelligence/Haas-ASR/src"` を付けて実行。

```bash
git clone https://github.com/third-intelligence/Haas-ASR ../Haas-ASR
haas/convert.sh          # Linux VM 側でも変換（または F32 GGUF をコピー）
haas/parity.sh           # F32: PASS が 2 つ出れば OK
haas/parity.sh third-intelligence/Haas-ja-v1.0 ~/haas-hex/models/Haas-ja-v1.0-Q8_0.gguf   # Q8_0 の差の確認
```

### 4. Hex をビルドしてインストール

```bash
haas/hex/update.sh
```

- 最新リリース（`vX.Y.Z` タグ）を使う。GGUF の取り込み機能（anomalyco/hex#759）は v3.3.0 から入っている。
  `--ref dev` などで別の ref も選べる（その ref が固定する transcribe.cpp が `haas/BASE` と違えば、Haas のコミットを自動で rebase する）。
- 中身: transcribe.cpp のビルドとテスト（初回は `samples/ja*.wav` の書き起こしを `~/haas-hex/golden/` に保存、以後はそれと比較）
  → Hex のビルド（`HEX_RELEASE_UNSIGNED=1 HEX_LOCAL_BUILD=1 scripts/build-app.sh`: 名前 Hex / bundle `anomaly.hex` / 更新チェックなし）
  → 「Hex Local」で署名し直し（`HexMicrophone.driver` → `scrawl-helper` → アプリ本体、公式の `sign-macos.sh` と同じ順序・entitlements）
  → `/Applications/Hex.app` を置き換えて起動。
- 初回だけ macOS がマイク・アクセシビリティ等の許可を改めて求めます（公式版と署名が違うため）。以後は同じ証明書で署名する限り維持されます。
- `/Applications` のアプリを置き換えるため、ターミナルに「App の管理」の許可を求められることがあります。

### 5. Hex で Haas を選ぶ

音声モデルの選択画面 → 下の **「ほかに N 個のモデル」** → 全モデル一覧のヘッダー右上 **「モデルファイルを追加…」**
→ `~/haas-hex/models/Haas-ja-v1.0-Q8_0.gguf`（ファイル選択で `Cmd+Shift+G` にパスを貼る）→ 追加された「Haas ja v1.0」を **使用**。

ターミナルからでも同じです（表示された `custom-…` の ID を `use` に渡す）:

```bash
/Applications/Hex.app/Contents/MacOS/hex model import ~/haas-hex/models/Haas-ja-v1.0-Q8_0.gguf --name "Haas ja v1.0"
/Applications/Hex.app/Contents/MacOS/hex model use <custom-… の ID>
```

2026-11-02 に M3 Mac で、この手順で Hex（dev, 3.2.0）に取り込み、日本語の音声入力ができることを確認済み（当時はリリースに取り込み機能がなく `--ref dev`）。

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

## Windows

2026-11-04 に Windows 11 x64（NVIDIA GPU のデスクトップ）で、この手順で音声入力まで確認済み。GPU なしのノート・ARM64 は未確認。

x64 版の Hex は音声認識の重い部分（エンコーダ）を Vulkan で動かします。独立 GPU がなくても、Intel / AMD の内蔵 GPU のドライバが
Vulkan（`C:\Windows\System32\vulkan-1.dll`）を提供していれば動きます。ARM64 版は CPU だけで動きます。予測ネット（KDA）はどちらでも CPU です。

### 1. 必要なものを入れる（PowerShell、初回のみ）

Hex 本家の `scripts/setup-windows.ps1` と同じ内容です。

```powershell
winget install --id Git.Git -e
winget install --id jqlang.jq -e
winget install --id Kitware.CMake -e
winget install --id Rustlang.Rustup -e
winget install --id Microsoft.VisualStudio.2022.BuildTools -e --override "--wait --passive --norestart --add Microsoft.VisualStudio.Workload.VCTools --add Microsoft.VisualStudio.Component.VC.Llvm.Clang --includeRecommended"
winget install --id KhronosGroup.VulkanSDK -e    # x64 のみ
winget install --id astral-sh.uv -e              # Windows でモデルを変換するときだけ
```

終わったら **ターミナルをすべて閉じて** 開き直し（Vulkan SDK が設定する `VULKAN_SDK` は新しいターミナルにしか入らない）、`rustup default stable`。

以下のコマンドは Git for Windows の **Git Bash**（スタートメニューの「Git Bash」、プロンプトに `MINGW64` と出る）で実行します。
PowerShell や、`C:\Windows\System32\bash.exe`（WSL の bash）では動きません。確認:

```bash
cargo --version && echo "$VULKAN_SDK" && uname -s   # cargo のバージョン、C:\VulkanSDK\…、MINGW64_NT-… が出れば OK
```

### 2. このリポジトリとモデルを置く（Git Bash）

```bash
git clone -c core.autocrlf=false https://github.com/raihalea-lab/transcribe.cpp-haas ~/haas-hex/transcribe.cpp
```

モデルは Mac の `~/haas-hex/models/Haas-ja-v1.0-Q8_0.gguf` を、Windows の `C:\Users\<ユーザー名>\haas-hex\models\` にコピー。
Mac に無ければ Windows で変換する（uv が必要。`.nemo` と PyTorch で数 GB のダウンロード）:

```bash
cd ~/haas-hex/transcribe.cpp && haas/convert.sh
```

次の手順の前にモデルを置いておくと、ビルド後に `samples/ja*.wav` の書き起こしも確認される。

### 3. ビルドしてインストール

- **x64**: Git Bash で

  ```bash
  cd ~/haas-hex/transcribe.cpp && haas/hex/update.sh
  ```

- **ARM64**: スタートメニューの「Developer PowerShell for VS 2022」で `& "C:\Program Files\Git\bin\bash.exe" -l` を実行してから同じコマンド
  （ggml の ARM 向けコードは MSVC を受け付けないので clang-cl を使う）。

初回は 30〜60 分ほどかかる見込み。中身は Mac と同じ流れで、最後に次を行います。

- `%LOCALAPPDATA%\Programs\Hex-Haas\hex.exe` に置き、スタートメニューに「Hex (Haas)」を作る
- 動いている Hex（公式版を含む）を終了させてから起動する
- 自動アップデートはオフ（`HEX_LOCAL_BUILD=1`）

公式の Hex が入っていると「アンインストールして」と表示されます。設定 › アプリ › Hex から外してください。設定・履歴・モデル（`%APPDATA%\hex`）は残ります。
公式インストーラは手元ビルドを入れられない（常に公式版をダウンロードする）ので、この形にしています。

Windows ではビルド後、transcribe.cpp の単体テストは飛ばし（本家も Windows では CI で回していない）、書き起こしの比較だけを行います。

日本語版の Windows で `transcribe-cli` を直接動かすと、書き起こしが文字化けして見えます（UTF-8 の出力をコンソールが
CP932 で表示するため。中身は正しい）。読める形で確かめるには、先に `chcp.com 65001` を実行するか、`-o out.txt` でファイルに書き出します。
Visual Studio のジェネレータでビルドした `transcribe-cli.exe` は `build/bin/Release/` にあります。

### 4. Hex で Haas を選ぶ

Mac と同じく、モデル選択 → 「ほかに N 個のモデル」→「モデルファイルを追加…」。またはコマンドで:

```bash
"$LOCALAPPDATA/Programs/Hex-Haas/hex.exe" model import ~/haas-hex/models/Haas-ja-v1.0-Q8_0.gguf --name "Haas ja v1.0"
"$LOCALAPPDATA/Programs/Hex-Haas/hex.exe" model use <表示された custom-… の ID>
```

ログイン時に起動したい場合は Hex の設定で「Launch at login」を ON（この exe が登録されます）。既定のショートカットは `Ctrl+Space`。

## メモ

- 予測ネットの重みは Q8_0 でもロード時に fp32 に展開されます（`A_log` `dt_bias` ノルム・畳み込みは GGUF 上も F32）。
  CER が ONNX 版（FLEURS ja 4.34%）より明らかに悪いときは、まず F32 GGUF と比べてください。
- Hex の語彙（vocabulary）は、TDT モデルでは書き起こし後のリスコアリングで、KDA でもそのまま動きます（合成モデルで動作確認のみ）。
- 自己署名だと画面収録用の仮想マイク（`HexMicrophone.driver`）は読み込まれない可能性があります（音声入力には不要）。
- 公式 Hex に戻すには、公式の DMG から入れ直します。

## ライセンスと出典

- このリポジトリ（Haas 対応の変更と `haas/` を含む）は、元の transcribe.cpp と同じ MIT License（ルートの `LICENSE`）。
- **Haas-ja-v1.0**（[third-intelligence/Haas-ja-v1.0](https://huggingface.co/third-intelligence/Haas-ja-v1.0)、Third Intelligence）は Apache-2.0。
  重みはここには含まず、`haas/convert.sh` が利用時に取得します。変換した GGUF（Apache-2.0 の派生物）を人に配る場合は、
  Apache-2.0 のライセンス文を添え、変換したことを明記してください（GGUF の `general.license` にも `apache-2.0` を書き込んでいます）。
- **Haas-ASR**（Apache-2.0）と **flash-linear-attention**（MIT）は、数値検証のときに実行時に読み込むだけで、同梱していません。
  `haas/kda_reference.py` で GPU 専用カーネルの代わりに使う短い関数は、flash-linear-attention 0.5.2 の実装に沿ったものです
  （MIT License, Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li）。
- **Hex**（MIT）のコードは含みません。`haas/hex/` のスクリプトは本家 Hex を取得し、その手順でビルドします。

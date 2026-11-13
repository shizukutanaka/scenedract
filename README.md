# SceneDract

Describe a scene in words. Get a verified blueprint, then working code for Blender and Unity.

SceneDract は、文章で頼んだ内容を「設計図（Scene IR）」に変換し、実行前に検証と自動修復を行ってから、Blender Python / Unity C# / Three.js プレビューを書き出すデスクトップアプリです。AI が書いたコードをそのまま実行するのではなく、設計図が残り、検証の証明書つきで DCC（Blender / Unity / Maya など）へ渡せることが特徴です。

## Download

Releases ページから、お使いの OS 向けのパッケージをダウンロードしてください。インストールは不要で、解凍してそのまま起動できます。

> **注意:** v1.0.0 はまだ公開されていません。リリースは現在下書きの状態で、パッケージの公開は後日行います。公開されるまでは、以下のパッケージをダウンロードすることはできません。

- Windows (x64): `SceneDract-v1.0.0-windows-x64.zip` → `SceneDract.exe`
- macOS (Apple Silicon): `SceneDract-v1.0.0-macos-arm64.zip` → `SceneDract.app`
- macOS (Intel): `SceneDract-v1.0.0-macos-x64.zip` → `SceneDract.app`
- Linux (x64): `SceneDract-v1.0.0-linux-x64.zip` → `./SceneDract`

署名・公証は行っていないため、初回起動時に OS の確認ダイアログが出る場合があります（各パッケージ同梱の README に回避手順を記載）。

## Updates

デスクトップ版は起動時に GitHub Releases を確認し（最大で1日1回まで。失敗した場合は何も表示しません）、新しいバージョンがあるとアプリ内にバナーを表示します。バナーからお使いの OS 向けの zip をアプリ内でダウンロードでき（進捗バー付き）、完了するとダウンロードしたファイルが表示されます。適用は全 OS 共通で手動です。zip を解凍して、今のアプリのフォルダと置き換えてください。自動でファイルを入れ替える仕組みは、未検証のまま配る危険を避けるため、意図的に入れていません。メニューの「更新を確認」から手動で確認することもできます。

## What it does

- 自然言語 → Scene IR（JSON の設計図）→ 検証・自動修復 → Blender / Unity 向けコード生成
- 検証: めり込み・浮遊・接地・支持関係に加え、重心と支持面による転倒リスク、張り出し、質量推定などの静的物理監査
- 書き出しプロファイル（GLB / FBX / USD など形式別の保持・喪失情報つき）
- VRChat の性能ランク目安、書き出し前プレビュー
- 42言語対応、初心者・標準・上級の3段階モード
- 自分の LLM エンドポイントを接続する BYO 方式にも対応（キーはアプリ内にのみ保持）

データは端末内にのみ保存され、外部に送信されません。外部へ接続するのは、BYO LLM を設定した場合のそのエンドポイントと、デスクトップ版の更新確認（GitHub へのバージョン照会のみで、ユーザーデータは送りません）だけです。

## Repository layout

- `app/` — アプリ本体（単一ファイルの Web アプリ `SceneDract.html` と、デスクトップ版のシェル `main.js`）
- Releases の各パッケージは、Electron 実行環境に `app/` を同梱したものです

## Status

Unity 実機での生成コードのコンパイル・実行、VR 実機での動作は未検証です。デスクトップ版の更新フローも、旧版から新版への実機ドリルは未実施です（バージョン比較とパッケージ選択のロジックは単体テストで確認済み）。検証できている範囲とできていない範囲は、アプリ内の検証証明書にも明記しています。

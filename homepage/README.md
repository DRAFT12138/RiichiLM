# RiichiLM homepage

[English](#english) · [简体中文](#简体中文) · [日本語](#日本語)

## English

This directory contains the standalone Chinese/English project homepage exported from the original Sites project. The live website is [www.riichilm.top](https://www.riichilm.top). It is separate from the React browser arena in `web/`.

From the repository root, preview it with Python 3:

```sh
python3 -m http.server 8000 --directory homepage
```

Open <http://localhost:8000>. No Node.js, build step, API keys, backend, or Sites account is required. Edit `index.html` directly; keep `data-zh` and `data-en` text equivalent.

For hosting, upload the contents of `homepage/` to a static host's document root. Set the publish directory to `homepage` and leave the build command empty. Relative asset paths also support a project subdirectory such as `/RiichiLM/`. For GitHub Pages, use a Pages workflow that uploads `homepage/`, or copy its contents to a dedicated publishing branch. This contribution does not configure a deployment, change DNS, or transfer the existing domain. Changes here do not automatically update the original Sites publication.

To export it, copy this directory together with the repository's `LICENSE` and `NOTICE`. Keep `index.html`, `favicon.svg`, and `assets/` together. External documentation links require internet access; there are no runtime CDN dependencies.

Assets: `assets/hero.png` is the AI-generated decorative illustration from the original Sites homepage. The lobby and spectator screenshots are copies of `assets/screenshots/` at the repository root; they show random opponents, not trained model performance. Preserve the repository's license and attribution notices when redistributing.

## 简体中文

本目录是从原 Sites 项目导出的中英双语项目主页，线上地址为 [www.riichilm.top](https://www.riichilm.top)。它独立于 `web/` 中的 React 对局界面。

在仓库根目录执行 `python3 -m http.server 8000 --directory homepage`，然后打开 <http://localhost:8000>。无需 Node.js、构建、API 密钥、后端或 Sites 账号。直接编辑 `index.html`，并保持 `data-zh` 与 `data-en` 文案一致。

部署时，将 `homepage/` 内的文件上传到静态托管根目录，发布目录设为 `homepage`，构建命令留空。资源使用相对路径，支持 `/RiichiLM/` 这样的子目录。GitHub Pages 可使用上传 `homepage/` 的工作流，或将其中的文件复制到专用发布分支。本次贡献不配置部署、不修改 DNS，也不迁移现有域名；仓库修改不会自动同步到原 Sites 网站。

导出时，复制本目录，并附带仓库根目录的 `LICENSE` 和 `NOTICE`；保持 `index.html`、`favicon.svg` 与 `assets/` 的相对位置。外部文档链接需要联网，页面本身没有运行时 CDN 依赖。

素材：`assets/hero.png` 为原 Sites 主页的 AI 生成装饰插画；大厅与观战截图复制自仓库根目录的 `assets/screenshots/`，使用随机对手，不代表训练模型的棋力。分发时保留仓库的许可与署名说明。

## 日本語

このディレクトリは、元の Sites プロジェクトから書き出した中国語・英語のプロジェクトホームページです。公開サイトは [www.riichilm.top](https://www.riichilm.top) です。`web/` の React 対局画面とは独立しています。

リポジトリのルートで `python3 -m http.server 8000 --directory homepage` を実行し、<http://localhost:8000> を開いてください。Node.js、ビルド、API キー、バックエンド、Sites アカウントは不要です。`index.html` を直接編集し、`data-zh` と `data-en` の内容を揃えてください。

静的ホスティングには `homepage/` 内のファイルをドキュメントルートへアップロードします。公開ディレクトリは `homepage`、ビルドコマンドは空にしてください。相対パスにより `/RiichiLM/` のようなサブディレクトリにも対応します。GitHub Pages では `homepage/` をアップロードするワークフロー、または専用公開ブランチへのコピーを利用できます。この変更はデプロイ、DNS の変更、既存ドメインの移行を設定しません。リポジトリの変更は元の Sites 公開サイトへ自動反映されません。

書き出す際は本ディレクトリと、リポジトリルートの `LICENSE`、`NOTICE` をコピーしてください。`index.html`、`favicon.svg`、`assets/` の相対位置を維持してください。外部ドキュメントへのリンクにはネット接続が必要ですが、ページ自体に実行時の CDN 依存はありません。

素材：`assets/hero.png` は元の Sites ホームページの AI 生成装飾イラストです。ロビーと観戦のスクリーンショットはリポジトリルートの `assets/screenshots/` からのコピーで、ランダム対戦相手を使用しており、学習済みモデルの強さを示すものではありません。再配布時はライセンスと帰属表示を保持してください。

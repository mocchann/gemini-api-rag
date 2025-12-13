# Gemini + Notion RAG

このリポジトリは、Notion ページのコンテンツを取得し、Gemini Embeddings と LangChain を使ってベクトル検索を行い、Gemini による回答生成を行う Retrieval Augmented Generation (RAG) パイプラインです。

## 1. セットアップ

1. **Python 環境の準備**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
2. **API キーと Notion Page ID を設定**
   ```bash
   cp .env.example .env
   # .env を開いて値を設定
   ```
   - `NOTION_API_KEY`: [Notion integration](https://www.notion.so/my-integrations) のシークレット
   - `NOTION_PAGE_ID`: 取り込みたいページの ID（URL 中の 32 文字程度の英数字）
   - `GOOGLE_API_KEY`: Gemini API のキー
   - `GEMINI_CHAT_MODEL` (任意): 回答生成に使うモデル ID。`models/xxx` 形式で指定（デフォルト: `models/gemini-3-pro-preview`）

## 2. 使い方

### (1) Notion データの取り込み
指定した Notion ページを取得し、チャンク化 → Embedding → ChromaDB へ保存します。既存のベクトルストアを作り直したい場合は `--reset` を付けてください。
```bash
python rag.py ingest
python rag.py ingest --reset  # 既存のベクトルストアを削除して再作成
```

### (2) 質問に回答
保存済みのベクトルストアを使って、ユーザー質問に回答します。
```bash
python rag.py ask "今週のタスクを教えて"
```

## 3. 処理の流れ
1. **Notion**: `notion-client` でターゲットページのブロック内容を取得。
2. **前処理**: LangChain の `RecursiveCharacterTextSplitter` でチャンク化。
3. **Embedding**: `GoogleGenerativeAIEmbeddings` (Gemini) でベクトル化。
4. **Vector DB**: `Chroma` に永続化。
5. **Retrieval**: ベクトル検索で関連チャンクを取得。
6. **Gemini**: `ChatGoogleGenerativeAI` とカスタムプロンプトで回答生成。

## 4. その他
- `python rag.py --help` でサブコマンドのヘルプが確認できます。
- 追加で前処理や Notion プロパティのマッピングが必要な場合は、`rag.py` 内の `build_document_from_page` をカスタマイズしてください。

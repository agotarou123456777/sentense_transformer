# install
ref : https://www.sbert.net/docs/installation.html

推奨依存関係
Python 3.8+
PyTorch 1.11.0+
transformers v4.34.0+

Default: モデルのロード、保存、推論 (= 埋め込みベクトルの取得) が可能
Default and Training: Default + トレーニングが可能
Development: 上記 + Sentence Transformers を開発するためのいくつかの依存関係を取得

→　基本的にはDefaultでよい. モデルをトレーニングするならDefault and Training. 

```
# Default
pip install -U sentence-transformers

# Default and Training
pip install -U "sentence-transformers[train]"
```

pretrained model
→　https://www.sbert.net/docs/sentence_transformer/pretrained_models.html

# quickstart
ref : https://www.sbert.net/docs/quickstart.html


## Sentence Transformer

Sentence Transformer (別名バイエンコーダー) モデルの特徴

1. text または image の固定サイズの埋め込みベクトル表現
2. 効率的な埋め込みベクトル計算・類似度計算
3. 幅広いタスク（意味的なテキストの類似性、意味的な検索、クラスタリング、分類、言い換えマイニングなど）に利用可能
4. として使用され 多くの場合、 2 ステップの取得プロセスの最初のステップ 、クロスエンコーダー (別名リランカー) モデルを使用して、バイエンコーダーからの上位 k 個の結果を再ランク付けします。


## Cross Encoder

Cross Encoder (別名リランカー) モデルの特徴:

1. テキストのペアの類似性スコアを計算
2. 一般に、Sentence Transformerモデルと比較して優れたパフォーマンス
3. テキストごとではなくペアごとに計算が必要なため、Sentence Transformer モデルよりも計算が遅くなる
4. 上位 k 位の結果を再ランク付けするためによく使用される

pretrained model
→　https://www.sbert.net/docs/cross_encoder/pretrained_models.html#
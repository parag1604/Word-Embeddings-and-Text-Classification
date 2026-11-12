# Word Embeddings and Text Classification

Notebooks on word embeddings and sentence (news) classification: exploring pretrained fastText vectors, then classifying AG News articles with an LSTM on learned embeddings, an LSTM on pretrained GloVe embeddings, and a transformer encoder.

---

## Theory

Lecture notes for this topic are coming soon on https://parag1604.github.io/notes

## Python Environment Setup

Use python >= 3.10 and install the requirements:

```bash
pip install -r requirements.txt
```

- PyTorch - models and training (install the hardware accelerated build if possible; the notebooks pick CUDA, then Apple MPS, then CPU)
- Numpy, Pandas - loading and preprocessing the data
- tqdm - progress bars
- Jupyter - to run the notebooks
- fasttext - only for notebook 1 (if `pip install fasttext` fails to build, try `pip install fasttext-wheel`)

### Pretrained embeddings

- Notebook 1 downloads the English fastText vectors itself (`fasttext.util.download_model('en')`, https://fasttext.cc/docs/en/crawl-vectors.html): `cc.en.300.bin.gz` is about 4.5 GB, unzips to about 7 GB in the working folder, and needs about 8 GB of RAM to load.
- Notebooks 3 and 4 need GloVe (https://nlp.stanford.edu/projects/glove/). Download `glove.6B.zip` (822 MB), unzip it, and put `glove.6B.100d.txt` in `./models/`.

## Data

AG News (Zhang, Zhao and LeCun, 2015, "Character-level Convolutional Networks for Text Classification"; built from the AG corpus, http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html): 120,000 training and 7,600 test news articles in 4 classes (World, Sports, Business, Sci/Tech). The commonly used CSV version (`ag_news_csv`, released with the paper) is mirrored in the CharCnn_Keras repository, https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv :

```bash
mkdir -p data/ag_news_csv
curl -o data/ag_news_csv/train.csv https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv
curl -o data/ag_news_csv/test.csv https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv
```

Each row is `class index (1-4), title, description`; `utils.py` joins title and description, lower-cases them and keeps only letters.

## Running

Start Jupyter in the repository folder (`jupyter notebook`) and run the notebooks in order. Each one starts with a cell saying what it teaches and what is new.

### 1. Word embeddings (`01_WordEmbeddings.ipynb`)

Loads the fastText vectors and explores the embedding space: nearest neighbours (`Bangalore` -> Chennai, Bengaluru, Pune, ...), vector arithmetic (`King - Man + Woman` -> `Queen`) and analogies (`berlin : germany :: paris : france`).

### 2. LSTM with learned embeddings (`02_SentenceClassification.ipynb`)

Builds a 20,000-word vocabulary, turns every article into 64 word ids (padded at the front), and trains an embedding layer + LSTM + linear layer from scratch for 5 epochs. Parag's run (an earlier version that dropped unknown words instead of mapping them to `<unk>`) reached 91.3% test accuracy (91.0% after the last epoch).

### 3. LSTM with frozen GloVe embeddings (`03_SentenceClassificationPretrained.ipynb`)

Same model, but the embedding layer is initialised with GloVe (100 dimensions, 400k words) and frozen. Parag's run (same earlier tokenization) reached 92.0% test accuracy.

### 4. Transformer encoder (`04_SentenceClassificationTransformers.ipynb`)

The LSTM is replaced by 3 transformer encoder layers on the frozen GloVe embeddings, with sinusoidal positional encodings, a padding mask, and mean pooling over the real tokens. Parag's run (an earlier version without positional encodings and with the padding included in the mean) reached 92.3% test accuracy.

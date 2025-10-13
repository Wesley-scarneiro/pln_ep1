import pandas as pd
import re
import unicodedata
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score

from corpusParams import CorpusParam

"""
    Trabalho de PLN — Classificação de textos
    Modelos:
        - DummyClassifier (baseline)
        - LogisticRegression (ajustada)
    Avaliação:
        - Acurácia média em 10 folds (validação cruzada)
        - Relatório de desempenho final
"""

# Aplica alguns tratamentos ao texto
def clean_text(text: str) -> str:
    def remove_accents(text: str) -> str:
        text = unicodedata.normalize('NFKD', text)
        return ''.join([c for c in text if not unicodedata.combining(c)])

    text = remove_accents(text)
    text = re.sub(r"[;:,.!?()\[\]\"'—–\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# Carrega o CSV e faz o tratamento básico dos textos
def read_csv(path: str, clean=False) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';')
    df.dropna(subset=['text'], inplace=True)
    df['text'] = df['text'].apply(str.lower)

    if clean:
        df['text'] = df['text'].apply(clean_text)
        
    return df

def dummy_classifier(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], 
        df['style'],
        test_size=0.3,
        random_state=42,
        stratify=df['style'])
    baseline_model = DummyClassifier(strategy="most_frequent")
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)

    print("--- Baseline ---")
    accuracy = accuracy_score(y_test, y_pred_baseline)
    print(f"Acurácia: {accuracy:.2f}")

# Definição dos modelos
models = {
    "LogisticRegression": LogisticRegression()
}

# Treina um modelo de LogisticRegression
def logistic_regression(corpus_list: list[CorpusParam]):
    for corpus in corpus_list:
        print("\n" + "=" * 70)
        print(f"Corpus atual: {corpus.name}")
        print("=" * 70)

        # Carrega os dados
        df = read_csv(corpus.path, clean=corpus.clean)
        X_text = df['text'].values 

        # Baseline
        dummy_classifier(df)

        # Codificação das classes
        le = LabelEncoder()
        y = le.fit_transform(df['style'])

        # Avaliação com 10-fold cross-validation 
        print("\n=== Validação Cruzada (10 folds) ===")
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        results = []

        # Treinamento do modelo
        print(f"\n--- LogisticRegression ---")
        model = LogisticRegression(
            C= corpus.logreg_C,
            penalty= corpus.penalty,
            solver= corpus.logreg_solver
        )
        scores = []
        for train_index, test_index in kfold.split(X_text, y):

            # Divisão dos dados de trieno e teste
            X_train_text, X_test_text = X_text[train_index], X_text[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Vetorização do texto
            vectorizer = TfidfVectorizer(
                sublinear_tf=corpus.sublinear, # True: resuz o impacto de palavras muito repetidas
                min_df=corpus.min_df, # Define a frequência mínima de um termo para ser incluído no vocabulário
                ngram_range=corpus.ngram_range, # Define a quantidade de n-gramas
                max_features=corpus.max_features # Limita o tamanho máximo do vocabulário
            )
            X_train_vec = vectorizer.fit_transform(X_train_text)
            X_test_vec = vectorizer.transform(X_test_text)
            
            # Treinar e avaliar o modelo
            model.fit(X_train_vec, y_train)
            score = model.score(X_test_vec, y_test)
            scores.append(score)

        mean_acc = np.mean(scores)
        std_acc = np.std(scores)
        print(f"Acurácia média (10 folds): {mean_acc:.4f} ± {std_acc:.4f}")
        results.append(('LogisticRegression', mean_acc, std_acc))
        
    return model

def main():
    # Arquivo de textos
    corpus_list = [
        CorpusParam(path='corpus/train_arcaico_moderno.csv', 
                    clean=False,
                    sublinear=True, 
                    min_df=3, 
                    ngram_range=(1, 2), 
                    max_features=60000,
                    logreg_c = 7.6178,
                    penalty = 'l2',
                    solver = 'saga'),
        CorpusParam(path='corpus/train_complexo_simples.csv', 
                    clean=False,
                    sublinear=True, 
                    min_df=4, 
                    ngram_range=(1, 3), 
                    max_features=None,
                    logreg_c = 7.7324,
                    penalty = 'l2',
                    solver = 'saga'),
        CorpusParam(path='corpus/train_literal_dinamico.csv', 
                    clean=False,
                    sublinear=True, 
                    min_df=1, 
                    ngram_range=(1, 3), 
                    max_features=50000,
                    logreg_c = 5.6427,
                    penalty = 'l2',
                    solver = 'saga')
    ]
    model = logistic_regression(corpus_list)

if __name__ == '__main__':
    main()
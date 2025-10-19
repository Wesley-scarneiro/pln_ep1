import pandas as pd
import re
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, accuracy_score
from corpusParams import CorpusParam
import numpy as np
import pickle

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
    df = pd.read_csv(path, sep=';', encoding='utf-8')
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
        
def run_logistic_regression(corpus_list: list['CorpusParam'], test_end=False) -> None:
    """
    Cria o modelo de Regressão Logística e o vetorizador TF-IDF com base nos parâmetros do corpus,
    treina o modelo com 70% dos dados e testa com 30%, exibindo o relatório de acurácia.
    Após isso, salva os modelos.
    """
    for corpus in corpus_list:
        print("\n" + "=" * 70)
        print(f"Corpus atual: {corpus.path}")
        print("=" * 70)

        # Carrega os dados
        df = read_csv(corpus.path, clean=corpus.clean)

        # Baseline
        dummy_classifier(df)

        X_text = df['text'].values
        y_text = df['style'].values

        # Codificação das classes
        le = LabelEncoder()
        y = le.fit_transform(y_text)

        # Divide os dados (70% treino / 30% teste)
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            X_text, y, test_size=0.3, random_state=42, stratify=y
        )

        # Vetorização do texto (ajuste apenas no treino)
        vectorizer = TfidfVectorizer(
            sublinear_tf=corpus.sublinear,
            min_df=corpus.min_df,
            ngram_range=corpus.ngram_range,
            max_features=corpus.max_features
        )
        X_train_vec = vectorizer.fit_transform(X_train_text)
        X_test_vec = vectorizer.transform(X_test_text)

        # Criação e treinamento do modelo
        model = LogisticRegression(
            C=corpus.logreg_C,
            penalty=corpus.penalty,
            solver=corpus.solver,
            random_state = 42
        )
        model.fit(X_train_vec, y_train)

        # Predição e relatório de desempenho
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        print(f"\nAcurácia (dados de teste - 30%): {acc:.4f}")
        print("\nRelatório por classe:")
        print(classification_report(y_test, y_pred, target_names=le.classes_, digits=4))

        # Realiza o teste final e salva resultados
        if (test_end):
            run_test_end(corpus, model, vectorizer, le)
            print(f'Processado arquivo de testes - {corpus.path_test}')
    

def crossval_accuracy(corpus_list: list['CorpusParam']):
    """
    Calcula a acurácia média em 10 folds para cada corpus usando Logistic Regression + TF-IDF.
    """
    for corpus in corpus_list:
        print("\n" + "="*70)
        print(f"Corpus atual: {corpus.path}")
        print("="*70)
        # Carrega os dados
        df = read_csv(corpus.path, clean=corpus.clean)
        X_text = df['text'].values
        y_text = df['style'].values

        # Codificação das classes
        le = LabelEncoder()
        y = le.fit_transform(y_text)

        # Definição da validação cruzada
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        scores = []

        for train_idx, test_idx in kfold.split(X_text, y):
            X_train_text, X_test_text = X_text[train_idx], X_text[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Vetorização
            vectorizer = TfidfVectorizer(
                sublinear_tf=corpus.sublinear,
                min_df=corpus.min_df,
                ngram_range=corpus.ngram_range,
                max_features=corpus.max_features
            )
            X_train_vec = vectorizer.fit_transform(X_train_text)
            X_test_vec = vectorizer.transform(X_test_text)

            # Modelo
            model = LogisticRegression(
                C=corpus.logreg_C,
                penalty=corpus.penalty,
                solver=corpus.solver,
            )
            model.fit(X_train_vec, y_train)

            # Avaliação
            y_pred = model.predict(X_test_vec)
            acc = accuracy_score(y_test, y_pred)
            scores.append(acc)

        mean_acc = np.mean(scores)
        std_acc = np.std(scores)
        print(f"Acurácia média (10 folds): {mean_acc:.4f} ± {std_acc:.4f}")

def run_test_end(corpus, model, vectorizer, le):
    df = read_csv(corpus.path_test, clean=corpus.clean)
    X_test_text = df['text'].values
    X_test_vec = vectorizer.transform(X_test_text)

    y_pred = model.predict(X_test_vec)
    y_labels = le.inverse_transform(y_pred)
    df_result = pd.DataFrame({
        'style': y_labels
    })
    df_result.to_csv(f'results/{corpus.corpus_name_test}.csv', index=False, encoding='utf-8')


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
                    solver = 'saga',
                    path_test= 'test/test_arcaico_moderno.csv'),
        CorpusParam(path='corpus/train_complexo_simples.csv', 
                    clean=False,
                    sublinear=True, 
                    min_df=4, 
                    ngram_range=(1, 3), 
                    max_features=None,
                    logreg_c = 7.7324,
                    penalty = 'l2',
                    solver = 'saga',
                    path_test= 'test/test_complexo_simples.csv'),
        CorpusParam(path='corpus/train_literal_dinamico.csv', 
                    clean=False,
                    sublinear=True, 
                    min_df=1, 
                    ngram_range=(1, 3), 
                    max_features=50000,
                    logreg_c = 5.6427,
                    penalty = 'l2',
                    solver = 'saga',
                    path_test= 'test/test_literal_dinamico.csv')
    ]
    #crossval_accuracy(corpus_list)
    run_logistic_regression(corpus_list, test_end=True)

if __name__ == '__main__':
    main()
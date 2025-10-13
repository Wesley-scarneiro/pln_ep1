import pandas as pd
import re
import unicodedata
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from scipy.stats import uniform, randint
from corpusParams import CorpusParam

def clean_text(text: str) -> str:
    """Remove acentos e pontuação básica."""
    def remove_accents(text: str) -> str:
        text = unicodedata.normalize('NFKD', text)
        return ''.join([c for c in text if not unicodedata.combining(c)])
    
    text = remove_accents(text)
    text = re.sub(r"[;:,.!?()\[\]\"'—–\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def read_csv(path: str, clean=False) -> pd.DataFrame:
    """Carrega CSV e limpa o texto se necessário."""
    df = pd.read_csv(path, sep=';')
    df.dropna(subset=['text'], inplace=True)
    df['text'] = df['text'].apply(str.lower)
    if clean:
        df['text'] = df['text'].apply(clean_text)
    return df

def randomized_search_tfidf_logreg(df: pd.DataFrame):
    """Executa RandomizedSearchCV para TF-IDF + Logistic Regression."""

    X_text = df['text'].values
    le = LabelEncoder()
    y = le.fit_transform(df['style'])

    # Pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('logreg', LogisticRegression(max_iter=1000))
    ])

    # Espaço de parâmetros
    param_dist = {
        # TF-IDF
        'tfidf__sublinear_tf': [True, False],
        'tfidf__min_df': randint(1, 5),
        'tfidf__ngram_range': [(1,1), (1,2), (1,3)],
        'tfidf__max_features': [5000, 10000, 20000, 30000, 40000, 50000, 60000, None],

        # Logistic Regression
        'logreg__C': uniform(0.01, 10),
        'logreg__penalty': ['l2'],
        'logreg__solver': ['lbfgs', 'saga']
    }

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_dist,
        n_iter=30,
        scoring='accuracy',
        n_jobs=-1,
        cv=kfold,
        verbose=2,
        random_state=42
    )

    random_search.fit(X_text, y)

    print("\n=== MELHORES PARÂMETROS ENCONTRADOS ===")
    print(random_search.best_params_)
    print(f"Acurácia média (CV): {random_search.best_score_:.4f}")

    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_

def run_gridsearch(corpus_list: list[tuple[str, bool, str]]):
    results = []
    for corpus in corpus_list:
        print("\n" + "="*70)
        print(f"Corpus atual: {corpus[2]}")
        print("="*70)

        df = read_csv(corpus[0], clean=corpus[1])
        best_model, best_params, best_score = randomized_search_tfidf_logreg(df)

        results.append({
            'corpus': corpus.name,
            'melhores_parametros': best_params,
            'acuracia_media': best_score,
            'clean': corpus.clean
        })

    # Salva resultados
    results_df = pd.DataFrame(results)
    results_df.to_csv("gridsearch.csv", index=False, sep=';')
    return results_df

def main():
    corpus_list = [
        ('corpus/train_arcaico_moderno.csv', False, 'train_arcaico_moderno'),
        ('corpus/train_arcaico_moderno.csv', True, 'train_arcaico_moderno'),
        ('corpus/train_complexo_simples.csv', False, 'train_complexo_simple'),
        ('corpus/train_complexo_simples.csv', True, 'train_complexo_simple'),
        ('corpus/train_literal_dinamico.csv', False, 'train_literal_dinamico'),
        ('corpus/train_literal_dinamico.csv', True, 'train_literal_dinamico')
    ]
    results_df = run_gridsearch(corpus_list)
    print("\n=== Resumo Final ===")
    print(results_df)

if __name__ == '__main__':
    main()

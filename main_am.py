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


"""
    Trabalho de PLN — Classificação de textos arcaicos vs modernos
    Modelos:
        - DummyClassifier (baseline)
        - LogisticRegression (ajustada)
    Avaliação:
        - Acurácia média em 10 folds (validação cruzada)
        - Relatório de desempenho final
"""


# --- Função para limpeza de texto ---
def clean_text(text: str) -> str:
    def remove_accents(text: str) -> str:
        text = unicodedata.normalize('NFKD', text)
        return ''.join([c for c in text if not unicodedata.combining(c)])

    text = str(text).lower()
    text = remove_accents(text)
    text = re.sub(r"[;:,.!?()\[\]\"'—–\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# --- Lê o CSV (ajuste o caminho para cada tarefa) ---
def read_csv(path: str, clean=False) -> pd.DataFrame:
    df = pd.read_csv(path, sep=';')
    if clean:
        df['text'] = df['text'].apply(clean_text)
    return df


# --- Lista de corpora ---
corpus = [
    'corpus/train_arcaico_moderno.csv',
    'corpus/train_complexo_simples.csv',
    'corpus/train_literal_dinamico.csv'
]


# --- Definição dos modelos ---
models = {
    "DummyClassifier": DummyClassifier(strategy="most_frequent"),
    "LogisticRegression": LogisticRegression(
        max_iter=2000,
        C=2.0,
        penalty='l2',
        n_jobs=-1
    )
}


# --- Loop principal sobre todos os corpus ---
for path in corpus:
    print("\n" + "=" * 70)
    print(f"📘 Corpus atual: {path}")
    print("=" * 70)

    # Carrega os dados
    df = read_csv(path, clean=True)

    # Codifica as classes
    le = LabelEncoder()
    y = le.fit_transform(df['style'])

    # Vetorização TF-IDF
    vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        max_df=0.80,
        min_df=3,
        ngram_range=(1, 2),
        max_features=10000
    )
    X = vectorizer.fit_transform(df['text'])

    # --- Avaliação com 10-fold cross-validation ---
    results = []
    print("\n=== Validação Cruzada (10 folds) ===")
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\n--- {name} ---")
        scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy', n_jobs=-1)
        mean_acc = np.mean(scores)
        std_acc = np.std(scores)

        print(f"Acurácia média (10 folds): {mean_acc:.4f} ± {std_acc:.4f}")
        results.append((name, mean_acc, std_acc))

    # Exibe comparativo
    df_results = pd.DataFrame(results, columns=["Modelo", "Acurácia Média", "Desvio Padrão"])
    df_results = df_results.sort_values(by="Acurácia Média", ascending=False).reset_index(drop=True)

    print("\n=== Comparativo Final de Modelos ===")
    print(df_results)

    # --- Treinamento final e relatório ---
    print("\n=== Relatório Final (treino/teste simples) ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    best_model = LogisticRegression(
        max_iter=2000,
        C=3.0,
        penalty='l2',
        solver='lbfgs',
        class_weight='balanced',
        n_jobs=-1
    )

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    print(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred)))
    final_acc = accuracy_score(y_test, y_pred)
    print(f"Acurácia final no conjunto de teste: {final_acc:.4f}")

print("\n🏁 Execução finalizada para todos os corpora.")

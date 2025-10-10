
======================================================================
📘 Corpus atual: corpus/train_arcaico_moderno.csv
======================================================================

=== Validação Cruzada (10 folds) ===

--- DummyClassifier ---
Acurácia média (10 folds): 0.4999 ± 0.0001

--- LogisticRegression ---
Acurácia média (10 folds): 0.8438 ± 0.0048

=== Comparativo Final de Modelos ===
               Modelo  Acurácia Média  Desvio Padrão
0  LogisticRegression        0.843834       0.004836
1     DummyClassifier        0.499946       0.000066

=== Relatório Final (treino/teste simples) ===
              precision    recall  f1-score   support

     arcaico       0.84      0.84      0.84      5533
     moderno       0.84      0.84      0.84      5533

    accuracy                           0.84     11066
   macro avg       0.84      0.84      0.84     11066
weighted avg       0.84      0.84      0.84     11066

Acurácia final no conjunto de teste: 0.8404

======================================================================
📘 Corpus atual: corpus/train_complexo_simples.csv
======================================================================

=== Validação Cruzada (10 folds) ===

--- DummyClassifier ---
Acurácia média (10 folds): 0.5000 ± 0.0001

--- LogisticRegression ---
Acurácia média (10 folds): 0.8385 ± 0.0072

=== Comparativo Final de Modelos ===
               Modelo  Acurácia Média  Desvio Padrão
0  LogisticRegression         0.83855       0.007232
1     DummyClassifier         0.49997       0.000060

=== Relatório Final (treino/teste simples) ===
              precision    recall  f1-score   support

    complexo       0.84      0.82      0.83      5014
     simples       0.83      0.85      0.84      5013

    accuracy                           0.84     10027
   macro avg       0.84      0.84      0.84     10027
weighted avg       0.84      0.84      0.84     10027

Acurácia final no conjunto de teste: 0.8365

======================================================================
📘 Corpus atual: corpus/train_literal_dinamico.csv
======================================================================

=== Validação Cruzada (10 folds) ===

--- DummyClassifier ---
Acurácia média (10 folds): 0.4999 ± 0.0001

--- LogisticRegression ---
Acurácia média (10 folds): 0.8409 ± 0.0045

=== Comparativo Final de Modelos ===
               Modelo  Acurácia Média  Desvio Padrão
0  LogisticRegression        0.840872       0.004521
1     DummyClassifier        0.499946       0.000066

=== Relatório Final (treino/teste simples) ===
              precision    recall  f1-score   support

    dinamico       0.83      0.84      0.84      5545
     literal       0.84      0.83      0.84      5545

    accuracy                           0.84     11090
   macro avg       0.84      0.84      0.84     11090
weighted avg       0.84      0.84      0.84     11090

Acurácia final no conjunto de teste: 0.8365

🏁 Execução finalizada para todos os corpora.
# RAG — Code du travail français (congés payés)

Démonstrateur **RAG (Retrieval-Augmented Generation)** complet et fonctionnel, appliqué à un
corpus juridique réel : les articles du Code du travail français relatifs aux **congés payés**
(articles L3141-1 à L3141-28).

## Pourquoi ce projet

Beaucoup de premiers projets RAG utilisent les documents personnels de leur auteur (CV, notes...).
Ce projet fait un choix différent : un corpus **public, vérifiable et réutilisable par n'importe
qui**, sur un sujet concret (le droit du travail), ce qui permet de juger objectivement la
pertinence des réponses en les comparant au texte de loi réel.

Aucune donnée confidentielle ou liée à un client n'est utilisée dans ce projet.

## Pipeline

```
Question en langage naturel
        │
        ▼
 ┌───────────────┐   TF-IDF + similarité cosinus (scikit-learn)
 │   Retrieval    │   → retrouve les k articles les plus pertinents
 └───────┬───────┘
         ▼
 ┌───────────────┐   Composition d'une réponse citée à partir
 │   Generation   │   du texte réel des articles récupérés
 └───────┬───────┘
         ▼
 Réponse + articles sources cités
```

1. **Corpus** (`data/corpus_conges_payes.json`) : 14 articles réels (L3141-1, 2, 3, 4, 5, 6, 8, 9,
   12, 13, 15, 19, 24, 28), chacun avec son identifiant, son texte intégral et son lien Légifrance.
2. **Retrieval** (`rag_pipeline.py::Retriever`) : vectorisation TF-IDF (bi-grammes, normalisation
   des accents) du corpus, puis similarité cosinus avec la question posée.
3. **Generation** (`rag_pipeline.py::Generator`) : la réponse est composée à partir du texte **réel**
   du ou des articles les plus pertinents, avec citation explicite de l'article. Ce mode
   "extractif" ne peut pas inventer de contenu juridique : il ne fait que restituer et citer.

## Transparence sur l'architecture (choix assumé)

Ce projet a été développé dans un environnement cloud dont la politique réseau n'autorise que
les registres de paquets (PyPI, npm...) et **bloque le téléchargement de poids de modèles**
depuis Hugging Face ou PyTorch Hub. Plutôt que d'annoncer un grand modèle de langage qu'il
n'était pas possible de réellement télécharger et tester dans cet environnement, le pipeline a
été conçu en deux couches explicitement interchangeables :

| Couche | Mode par défaut (100% local, sans téléchargement) | Mode neuronal (optionnel) |
|---|---|---|
| Retrieval | TF-IDF + cosinus (scikit-learn) | Embeddings denses multilingues (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`) |
| Generation | Composition extractive citée (sans invention possible) | Reformulation par un modèle texte-à-texte local (ex. `google/flan-t5-base`) |

Le code détecte automatiquement ce qui est disponible : si `sentence-transformers` ou
`transformers` sont installés et que les modèles sont accessibles (ce qui est le cas sur une
machine avec accès internet standard à Hugging Face), il suffit de passer
`embedding_backend="sentence-transformers"` et/ou `generation_backend="transformers"` à
`LegalRag(...)` pour activer le mode neuronal complet — avec repli automatique et message clair
si le modèle n'est pas disponible.

```python
# Mode 100% local (par défaut)
rag = LegalRag("data/corpus_conges_payes.json")

# Mode neuronal complet (sur une machine avec accès à Hugging Face)
rag = LegalRag(
    "data/corpus_conges_payes.json",
    embedding_backend="sentence-transformers",
    generation_backend="transformers",
)
```

## Installation

```bash
pip install -r requirements.txt
```

Pour le mode neuronal complet, installer en plus :

```bash
pip install sentence-transformers transformers torch
```

## Utilisation

```bash
python rag_pipeline.py
```

Ou voir le notebook `RAG_Code_du_travail_conges_payes.ipynb` pour une démonstration commentée
avec plusieurs exemples de questions/réponses et une analyse des limites du retrieval par
mots-clés.

## Exemple

```
Q: Un salarié peut-il prendre ses congés dès son embauche ?
Réponse: D'après l'Article L3141-12 du Code du travail : « Les congés peuvent être pris dès
l'embauche, sans préjudice des règles de détermination de la période de prise des congés et de
l'ordre des départs et des règles de fractionnement du congé fixées dans les conditions prévues
à la présente section. » (voir aussi : L3141-6, L3141-28).
```

## Limites connues

- Le corpus est volontairement restreint à la section congés payés (et non l'ensemble du Code
  du travail), pour rester manipulable et vérifiable.
- Le retrieval par mots-clés (TF-IDF) peut manquer des liens sémantiques entre une question et
  un article qui ne partage pas de vocabulaire explicite (exemple détaillé dans le notebook) —
  c'est la motivation principale pour le mode neuronal optionnel décrit ci-dessus.
- Ce projet est un démonstrateur pédagogique et ne constitue pas un conseil juridique.

## Pistes d'amélioration

- Étendre le corpus à d'autres sections du Code du travail.
- Activer le mode neuronal complet (embeddings denses + modèle de génération local).
- Ajouter une évaluation quantitative (précision du retrieval sur un jeu de questions annotées).
- Exposer le pipeline via une API (FastAPI) ou une interface (Streamlit).

## Stack technique

Python, scikit-learn (TF-IDF, similarité cosinus), NumPy — et, en option, sentence-transformers
et transformers (Hugging Face) pour le mode neuronal complet.

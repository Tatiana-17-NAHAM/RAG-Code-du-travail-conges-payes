"""
RAG (Retrieval-Augmented Generation) sur le Code du travail français — congés payés
=====================================================================================

Projet pédagogique démontrant un pipeline RAG complet et fonctionnel :
  1. Un corpus réel (14 articles du Code du travail français, section congés payés,
     articles L3141-1 à L3141-28, récupérés depuis des sources publiques).
  2. Une étape de récupération (retrieval) : vectorisation du corpus + recherche
     par similarité pour retrouver les articles pertinents à une question.
  3. Une étape de génération : composition d'une réponse en langage naturel,
     strictement ancrée (groundée) dans les articles récupérés, avec citation
     de la source légale.

Choix d'architecture et transparence
-------------------------------------
Ce pipeline a été développé et testé dans un environnement cloud dont la
politique réseau n'autorise que les registres de paquets (PyPI, npm...) et
bloque le téléchargement de poids de modèles pré-entraînés depuis Hugging Face
ou PyTorch Hub. Plutôt que de prétendre faire tourner un grand modèle de
langage sans pouvoir réellement le télécharger ni le tester, ce module est
construit en deux couches explicitement interchangeables :

  - EMBEDDING_BACKEND :
      "tfidf"  (par défaut, 100% local, aucun téléchargement) -> scikit-learn
      "sentence-transformers" -> si disponible (ex: en exécutant ce code sur une
      machine avec accès à Hugging Face), utilise un modèle multilingue
      (paraphrase-multilingual-MiniLM-L12-v2) pour des embeddings denses.

  - GENERATION_BACKEND :
      "extractive" (par défaut) -> compose la réponse à partir du texte réel
      des articles récupérés (aucune hallucination possible : la réponse ne
      contient que du texte juridique vérifié).
      "transformers" -> si un modèle texte-à-texte local est disponible
      (ex: google/flan-t5-base), l'utilise pour reformuler la réponse à partir
      du contexte récupéré.

Le code détecte automatiquement ce qui est disponible et bascule sur le mode
local si le mode avancé n'est pas accessible, en l'indiquant clairement dans
les logs. Voir le README pour la procédure d'activation du mode neuronal complet.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

EmbeddingBackend = Literal["tfidf", "sentence-transformers"]
GenerationBackend = Literal["extractive", "transformers"]


def normalize_fr(text: str) -> str:
    """Normalisation simple pour le français : minuscule + suppression des accents
    pour la vectorisation (améliore le rappel sur des variantes orthographiques)."""
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text


@dataclass
class Article:
    id: str
    titre: str
    texte: str
    source: str


@dataclass
class RetrievedArticle:
    article: Article
    score: float


@dataclass
class RagAnswer:
    question: str
    reponse: str
    articles_cites: list[RetrievedArticle] = field(default_factory=list)
    backend_embedding: str = "tfidf"
    backend_generation: str = "extractive"


class LegalCorpus:
    """Charge et expose le corpus d'articles du Code du travail."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        with open(self.path, encoding="utf-8") as f:
            raw = json.load(f)
        self.articles: list[Article] = [Article(**a) for a in raw]

    def __len__(self) -> int:
        return len(self.articles)

    def __iter__(self):
        return iter(self.articles)


class Retriever:
    """Étape de récupération : retrouve les articles les plus pertinents pour
    une question donnée, parmi le corpus.

    Backend par défaut : TF-IDF + similarité cosinus (scikit-learn), 100% local.
    Backend avancé optionnel : embeddings denses via sentence-transformers, avec
    repli automatique sur TF-IDF si le modèle n'est pas disponible.
    """

    def __init__(self, corpus: LegalCorpus, backend: EmbeddingBackend = "tfidf"):
        self.corpus = corpus
        self.requested_backend = backend
        self.backend: EmbeddingBackend = "tfidf"
        self._st_model = None

        self._texts = [f"{a.titre}. {a.texte}" for a in corpus]
        self._texts_norm = [normalize_fr(t) for t in self._texts]

        # TF-IDF : toujours construit, utilisé comme backend par défaut et comme
        # filet de sécurité si le backend neuronal n'est pas disponible.
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self._tfidf_matrix = self._vectorizer.fit_transform(self._texts_norm)

        if backend == "sentence-transformers":
            self._try_load_sentence_transformers()

    def _try_load_sentence_transformers(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
            self._st_embeddings = model.encode(self._texts, normalize_embeddings=True)
            self._st_model = model
            self.backend = "sentence-transformers"
            print("[Retriever] Backend neuronal 'sentence-transformers' chargé avec succès.")
        except Exception as exc:  # noqa: BLE001 - on veut capturer tout type d'échec réseau/import
            print(
                "[Retriever] Backend 'sentence-transformers' indisponible "
                f"({exc.__class__.__name__}: {exc}). Repli sur TF-IDF (100% local)."
            )
            self.backend = "tfidf"

    def retrieve(self, query: str, k: int = 3) -> list[RetrievedArticle]:
        if self.backend == "sentence-transformers" and self._st_model is not None:
            q_emb = self._st_model.encode([query], normalize_embeddings=True)
            sims = cosine_similarity(q_emb, self._st_embeddings)[0]
        else:
            q_vec = self._vectorizer.transform([normalize_fr(query)])
            sims = cosine_similarity(q_vec, self._tfidf_matrix)[0]

        order = np.argsort(sims)[::-1][:k]
        return [
            RetrievedArticle(article=self.corpus.articles[i], score=float(sims[i]))
            for i in order
            if sims[i] > 0
        ]


class Generator:
    """Étape de génération : compose une réponse en langage naturel à partir
    des articles récupérés.

    Backend par défaut : "extractive" — assemble la réponse strictement à
    partir du texte réel des articles (aucune invention possible).
    Backend avancé optionnel : "transformers" — reformule le contexte récupéré
    avec un modèle texte-à-texte local (ex: flan-t5-base), avec repli
    automatique sur le mode extractif si aucun modèle n'est disponible.
    """

    def __init__(self, backend: GenerationBackend = "extractive"):
        self.requested_backend = backend
        self.backend: GenerationBackend = "extractive"
        self._pipeline = None

        if backend == "transformers":
            self._try_load_transformers()

    def _try_load_transformers(self) -> None:
        try:
            from transformers import pipeline  # type: ignore

            self._pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
            self.backend = "transformers"
            print("[Generator] Backend neuronal 'transformers' (flan-t5-base) chargé avec succès.")
        except Exception as exc:  # noqa: BLE001
            print(
                "[Generator] Backend 'transformers' indisponible "
                f"({exc.__class__.__name__}: {exc}). Repli sur le mode extractif."
            )
            self.backend = "extractive"

    def generate(self, question: str, retrieved: list[RetrievedArticle]) -> str:
        if not retrieved:
            return (
                "Je n'ai trouvé aucun article du corpus (congés payés, "
                "Code du travail) répondant directement à cette question."
            )

        if self.backend == "transformers" and self._pipeline is not None:
            context = "\n".join(f"{r.article.titre} : {r.article.texte}" for r in retrieved)
            prompt = (
                "Réponds en français à la question suivante en te basant "
                f"uniquement sur le contexte juridique donné.\n\nContexte:\n{context}"
                f"\n\nQuestion: {question}\nRéponse:"
            )
            out = self._pipeline(prompt, max_new_tokens=160)[0]["generated_text"]
            return out.strip()

        # Mode extractif par défaut : réponse construite et citée, sans invention.
        top = retrieved[0]
        extrait = top.article.texte
        if len(retrieved) == 1:
            return (
                f"D'après l'{top.article.titre} du Code du travail : « {extrait} »"
            )
        autres = ", ".join(r.article.titre.replace("Article ", "") for r in retrieved[1:])
        return (
            f"D'après l'{top.article.titre} du Code du travail : « {extrait} » "
            f"(voir aussi : {autres})."
        )


class LegalRag:
    """Pipeline RAG complet : question -> articles pertinents -> réponse citée."""

    def __init__(
        self,
        corpus_path: str | Path,
        embedding_backend: EmbeddingBackend = "tfidf",
        generation_backend: GenerationBackend = "extractive",
    ):
        self.corpus = LegalCorpus(corpus_path)
        self.retriever = Retriever(self.corpus, backend=embedding_backend)
        self.generator = Generator(backend=generation_backend)

    def ask(self, question: str, k: int = 3) -> RagAnswer:
        retrieved = self.retriever.retrieve(question, k=k)
        reponse = self.generator.generate(question, retrieved)
        return RagAnswer(
            question=question,
            reponse=reponse,
            articles_cites=retrieved,
            backend_embedding=self.retriever.backend,
            backend_generation=self.generator.backend,
        )


if __name__ == "__main__":
    DATA_PATH = Path(__file__).parent / "data_corpus_conges_payes.json"
    rag = LegalRag(DATA_PATH)

    questions = [
        "Combien de jours de congés payés acquiert-on par mois de travail ?",
        "Un salarié peut-il prendre ses congés dès son embauche ?",
        "Que se passe-t-il pour les congés payés si le contrat de travail est rompu ?",
        "Comment est fixé l'ordre des départs en congés ?",
        "Les congés payés comptent-ils comme du temps de travail effectif ?",
    ]

    for q in questions:
        result = rag.ask(q, k=3)
        print("=" * 90)
        print("Q:", result.question)
        print("-" * 90)
        print("Réponse:", result.reponse)
        print("Articles récupérés:")
        for r in result.articles_cites:
            print(f"  - {r.article.titre} (score={r.score:.3f})")

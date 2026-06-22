# Rag-on-EU-Taxonomy

Petit RAG d'entraînement sur les FAQ de l'EU Taxonomy. L'objectif est de concevoir et implémenter un chatbot simple capable de retrouver des informations pertinentes sur l'EU Taxonomy.

## Prérequis

- **Python 3.10, 3.11 ou 3.12** (recommandé : 3.11)
- **Git**
- Connexion internet pour le premier lancement (téléchargement des modèles d'embedding et, optionnellement, du modèle NLI)

Aucune clé API n'est requise pour démarrer l'application. Les onglets Benchmark, Interactive test et Data explorer fonctionnent sans LLM.

## Installation locale

### Windows (PowerShell)

```powershell
git clone <url-du-repo>
cd "RAG - Implementation"

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[ui]"
eu-taxonomy-rag
```

Si `py -3.11` n'est pas reconnu, installez Python depuis [python.org](https://www.python.org/downloads/) en cochant **Add Python to PATH**, puis relancez les commandes ci-dessus.

### macOS

```bash
git clone <url-du-repo>
cd "RAG - Implementation"

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[ui]"
eu-taxonomy-rag
```

Sur macOS avec Homebrew : `brew install python@3.11` si besoin.

### Linux

```bash
git clone <url-du-repo>
cd "RAG - Implementation"

python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[ui]"
eu-taxonomy-rag
```

Sur Debian/Ubuntu : `sudo apt install python3.11 python3.11-venv` si besoin.

### macOS / Linux — une seule commande

```bash
chmod +x scripts/start.sh
./scripts/start.sh
```

Ce script installe le package en mode éditable avec l'interface Streamlit, puis lance l'application.

## Installation Docker

Le dépôt inclut un `Dockerfile` et un `docker-compose.yml`.

### Avec Docker Compose (recommandé)

```bash
git clone <url-du-repo>
cd "RAG - Implementation"

# Optionnel : clés LLM pour l'onglet Chatbot
cp .env.example .env
# éditez .env si nécessaire
# puis décommentez env_file dans docker-compose.yml

docker compose up --build
```

Ouvrez [http://localhost:8501](http://localhost:8501).

Le cache (chunks, index de recherche, base d'évaluation) est persisté via un **bind mount** sur `./.cache` du dépôt. Les modèles Hugging Face sont mis en cache dans le volume Docker `eu-taxonomy-rag_hf-cache`.

Le build Docker met en cache l'installation `pip` : elle n'est relancée que si `pyproject.toml` ou `README.md` change. Une modification du code (`src/`, `app/`, etc.) recompile uniquement les couches suivantes (quelques secondes).

### Avec Docker seul

```bash
docker build -t eu-taxonomy-rag .
docker run --rm -p 8501:8501 \
  -e EU_TAXONOMY_PROJECT_ROOT=/app \
  -v "$(pwd)/.cache:/app/.cache" \
  -v eu-taxonomy-rag-hf-cache:/root/.cache/huggingface \
  eu-taxonomy-rag
```

> **Important :** montez le volume sur le répertoire `.cache`, pas sur le fichier `generation_eval.db`. Monter un répertoire vide à la place du fichier SQLite empêche l'ouverture de la base.

## Premier lancement

Quelle que soit la méthode d'installation, le lanceur :

1. Construit les **chunks** FAQ à partir de `data/taxonomy_faqs_cleaned.md` (mis en cache dans `.cache/chunks.jsonl`)
2. Initialise la base SQLite d'**évaluation de génération** (`.cache/generation_eval.db`, créée à la première utilisation du chatbot)
3. Ouvre le **tableau de bord Streamlit** dans le navigateur

Ensuite, ouvrez l'onglet **Benchmark** et cliquez sur **Build indexes** pour créer les index de recherche (BM25 + dense). Cette étape télécharge les modèles d'embedding et peut prendre plusieurs minutes ; les lancements suivants réutilisent `.cache/index/`.

### Commandes utiles

```bash
eu-taxonomy-rag --bootstrap-only      # prépare les chunks sans ouvrir l'UI
eu-taxonomy-rag --force-rebuild       # reconstruit les chunks depuis le fichier FAQ
```

## Clés LLM (onglet Chatbot, optionnel)

Pour l'onglet **Chatbot**, fournissez une clé API via l'une des options suivantes :

1. Copier le modèle et renseigner vos clés :

   ```bash
   cp .env.example .env
   ```

2. Saisir les identifiants dans l'interface et cliquer sur **Save credentials to .env** (le fichier `.env` est créé à la première sauvegarde).

Providers supportés : OpenAI, Azure OpenAI, AWS Bedrock, API compatible OpenAI.

## Dépendances optionnelles

```bash
pip install -e ".[ui,faiss,dev]"
```

| Extra | Contenu |
|-------|---------|
| `ui` | Streamlit, pandas, matplotlib (inclus dans les instructions ci-dessus) |
| `faiss` | Index dense FAISS (sinon index NumPy par défaut) |
| `dev` | pytest pour les tests |

## Pages Streamlit

| Page | Rôle |
|------|------|
| **Chatbot** | Q&R RAG + évaluation de groundedness |
| **Benchmark** | Évaluation retrieval multi-jeux (Recall@K, MRR) |
| **Interactive test** | Comparaison côte à côte des méthodes de retrieval |
| **Data explorer** | Exploration des chunks et jeux d'évaluation |
| **Saved results** | Comparaison des exports JSON de benchmark |

## Évaluation de génération (groundedness)

Le chatbot peut évaluer chaque réponse générée pour sa **fidélité au contexte** (groundedness / faithfulness) par rapport aux chunks FAQ récupérés.

### Déroulement

Après la génération LLM, l'application :

1. Découpe la réponse en affirmations courtes.
2. Compare chaque affirmation aux chunks récupérés avec un modèle NLI léger (`typeform/distilbert-base-uncased-mnli`).
3. Étiquette chaque affirmation : `supported`, `contradicted` ou `not_enough_info`.
4. Enregistre l'interaction dans SQLite (`.cache/generation_eval.db`).
5. Affiche les métriques dans **Chat**, **History** et **Metrics**.

### Métriques

| Métrique | Signification |
|----------|---------------|
| **Faithfulness** | `supported_claims / total_claims` |
| **Contradiction rate** | Part des affirmations contredites |
| **Unsupported rate** | Part des affirmations sans information suffisante |
| **Best / average claim score** | Score d'entailment max et moyen par affirmation |
| **Score range** | Écart entre le meilleur et le pire score |

### Activer / désactiver

Dans `.env` ou les variables d'environnement :

```bash
ENABLE_GENERATION_EVAL=true   # défaut
ENABLE_GENERATION_EVAL=false  # désactive l'évaluation NLI et les écritures SQLite
```

### Limites

- Outil de **diagnostic**, pas un juge automatique parfait.
- Le modèle NLI peut se tromper sur les paraphrases ou le vocabulaire métier.
- Le découpage en affirmations est basé sur les phrases.
- Les scores dépendent de la qualité du retrieval.
- La première évaluation télécharge le modèle NLI (quelques secondes sur CPU).

## Variables d'environnement

| Variable | Description | Défaut |
|----------|-------------|--------|
| `EU_TAXONOMY_PROJECT_ROOT` | Racine du projet (données, cache) | Détection auto (cwd ou emplacement du package) |
| `EU_TAXONOMY_EVAL_DB` | Chemin de la base SQLite d'évaluation | `<PROJECT_ROOT>/.cache/generation_eval.db` |
| `ENABLE_GENERATION_EVAL` | Active l'évaluation groundedness | `true` |
| `EU_TAXONOMY_LLM_PROVIDER` | Fournisseur LLM du chatbot | — |
| `EU_TAXONOMY_LLM_MODEL` | Modèle LLM | `gpt-4o-mini` |
| `OPENAI_API_KEY` | Clé OpenAI | — |

Voir `.env.example` pour la liste complète des paramètres chatbot.

## Arborescence

```
app/
  streamlit_app.py       # tableau de bord principal
  chatbot_page.py        # chatbot RAG
  generation_eval_ui.py  # UI d'évaluation groundedness
src/eu_taxonomy_rag/
  cli.py                 # bootstrap + lancement Streamlit
  paths.py               # chemins projet et cache
  evaluation/            # benchmarks, métriques, NLI
  storage/               # persistance SQLite
data/
  taxonomy_faqs_cleaned.md
  evaluation/            # jeux golden et résultats exportés
.cache/                  # chunks, index, base d'évaluation (généré localement)
```

## Développement

```bash
pip install -e ".[ui,dev]"
pytest
eu-taxonomy-rag --bootstrap-only
```

## Dépannage

| Problème | Piste de résolution |
|----------|---------------------|
| `unable to open database file` | Vérifiez les droits d'écriture sur `.cache` ; en Docker, montez un volume sur `.cache` (pas sur le fichier `.db`) |
| FAQ introuvable en Docker | Définissez `EU_TAXONOMY_PROJECT_ROOT=/app` |
| `sentence-transformers` / torch | Utilisez Python 3.10–3.12 ; évitez 3.13+ |
| Port 8501 occupé | `docker compose` : changez `8501:8501` ; local : Streamlit propose un autre port |

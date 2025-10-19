# Odia-TriGCN

**Odia-TriGCN** is a triplet extraction framework designed for the low-resource **Odia** language using a Customized Dependency Parser. It performs **Aspect Sentiment Triplet Extraction (ASTE)** by jointly leveraging contextual, syntactic, and auxiliary linguistic signals to identify structured *(Aspect, Opinion, Sentiment)* triplets from Odia text.

## Model Overview
Odia-TriGCN integrates several complementary components:
- **Multilingual Contextual Encoder (mBERT):** captures deep contextual and semantic representations across languages.
- **Customized Dependency Parser:** a linguistically grounded parser trained on manually annotated Odia Universal Dependencies (UD) data, providing accurate syntactic relations for graph construction.
- **Tri-Channel Graph Convolutional Network (GCN):** models syntactic dependencies, tree-based distances, and biaffine semantic attention for fine-grained structural reasoning.
- **Auxiliary Feature Integration:** incorporates sentiment and emotion cues from an Odia news corpus, and employs cross-lingual supervision using an Odia–English parallel corpus to improve generalization.
- **Grid-Tagging Representation:** enables structured decoding of aspect–opinion–sentiment triplets within a unified architecture.

## Dataset and Training
The model is trained and evaluated on a manually curated **Odia-Triplet** dataset, annotated for aspect, opinion, and sentiment across multiple domains. This dataset enables fine-grained extraction of relational triplets in Odia reviews and opinion texts.
## Data Availability
A subset of the dataset is currently provided for research and reproducibility. The complete dataset will be released publicly following the paper’s acceptance.

---



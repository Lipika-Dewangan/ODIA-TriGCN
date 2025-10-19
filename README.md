# ODIA-TriGCN
Odia-TriGCN is a triplet extraction framework designed for the low-resource Odia language. It performs Aspect Sentiment Triplet Extraction (ASTE) by jointly leveraging contextual, syntactic, and auxiliary signals to identify structured (Aspect, Opinion, Sentiment) triplets from Odia text.
Odia-TriGCN combines multiple complementary components:

Multilingual Contextual Encoder (mBERT): captures deep semantic and contextual representations.

Tri-Channel Graph Convolutional Network (GCN): models syntactic relations, tree-based distances, and biaffine semantic attention for fine-grained structural reasoning.

Auxiliary Feature Integration: incorporates sentiment- and emotion-rich cues from an Odia news corpus and cross-lingual supervision using an Odia–English parallel corpus to enhance contextual generalization.

Grid-Tagging Representation: enables structured decoding of triplets within a unified framework.

The model is trained and evaluated on a manually curated Odia-Triplet dataset, enabling the extraction of structured (Aspect, Opinion, Sentiment) triplets from Odia text.

A subset of the dataset is currently provided. The full dataset will be released upon acceptance of the paper.


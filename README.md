# ODIA-TriGCN
Odia-TriGCN is a triplet extraction model tailored for the low-resource Odia language, designed to perform Aspect Sentiment Triplet Extraction (ASTE) by leveraging both contextual and syntactic signals. Odia-TriGCN integrates:
A multilingual contextual encoder (mBERT),

A Tri-Channel Graph Convolutional Network capturing syntactic relations, tree-based distances, and biaffine attention,

A Grid Tagging representation for structured triplet decoding.

The model is trained and evaluated on a manually curated Odia-Triplet dataset, enabling the extraction of structured (Aspect, Opinion, Sentiment) triplets from Odia text.

A subset of the dataset is currently provided. The full dataset will be released upon acceptance of the paper.


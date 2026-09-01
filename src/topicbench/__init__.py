"""TopicBench: benchmarking LLMs on labeling latent topic keywords.

Importing this package does not touch the network. The embedding model used by
the ``calculate_embedding_*`` functions is loaded lazily and cached on first use
-- see ``semantic_similarity.get_model``.
"""

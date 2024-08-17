# Russian Neural Topic Model based on Variational AutoEncoder
Basic NTM based on VAE from [article](https://arxiv.org/pdf/2401.15351).

# Model arhicture
 - [Text normalization](src/data/normalizer.py#L7);
 - [Variational AutoEncoder](src/model/vae.py#L5);

# Model results
 - method **topic_distribution(X: Tensor)** return $\theta$ - probabilities of topics
 - method **sample_words(X: Tensor, num_words: int)** return indexes of topic's words in vocabulary

Q1. 
The default embedding dimension of `text-embedding-3-small` is 1536, as noted above. 
1. Is there any way to modify this dimension?
2. What technique does OpenAI use to achieve this?

A1. 
1. Yes, you can modify this dimension by changing the dimensions parameter in self.client.embeddings.create(input=batch, model=self.embeddings_model_name, dimensions=N), defined in aimakerspace.openai_utils.embedding.EmbeddingModel)
2. The technique used is Matryoshka Representation Learning (MRL). It works by nesting information, storing the most important semantic information in the earlier dimensions and adding finer details in later dimensions. https://arxiv.org/abs/2205.13147


Q2. 
What are the benefits of using an `async` approach to collecting our embeddings?

A2. 
The key benefit of using async is to improve performance by allowing processes to execute concurrently (asynchronously) rather than sequentially (synchronously). There is no need for sequential processing of collecting embeddings because these are independent lookup operations. This therefore allows for massive scalability, enabling the program to handle orders of magnitude more embedding requests.


Q3. 
When calling the OpenAI API - are there any ways we can achieve more reproducible outputs?

A3.
There are several ways:
1. Set the seed parameter to a value and persist that value in future sessions. While not 100% reproducible, it will be much more deterministic than without a seed value.
2. Set the temperature = 0. Higher temperature is more creative/random, lower temperature is more generic, technical or structured, therefore more likely to produce a reproducible response.


Q4. What prompting strategies could you use to make the LLM have a more thoughtful, detailed response?
What is that strategy called?

A4. Chain of Thought (CoT) prompting is a strategy that can be used to make the LLM have a more thoughtful, detailed response by making the LLM iterate through each step, forcing the model to break down a complex problem into discretized pieces, which has been shown to produce more accurate and thoughtful responses, particularly for reasoning steps.
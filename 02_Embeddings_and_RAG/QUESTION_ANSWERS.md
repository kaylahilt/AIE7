Q1. 
The default embedding dimension of `text-embedding-3-small` is 1536, as noted above. 
1. Is there any way to modify this dimension?
2. What technique does OpenAI use to achieve this?
> NOTE: Check out this [API documentation](https://platform.openai.com/docs/api-reference/embeddings/create) for the answer to question #1, and [this documentation](https://platform.openai.com/docs/guides/embeddings/use-cases) for an answer to question #2!

A1. 
1. Yes, you can modify this dimension by changing the dimensions parameter in self.client.embeddings.create(input=batch, model=self.embeddings_model_name, dimensions=N), defined in aimakerspace.openai_utils.embedding.EmbeddingModel.
2. The technique used is Matryoshka Representation Learning (MRL). It works by nesting information, storing the most important semantic information in the earlier dimensions and adding finer details in later dimensions. https://arxiv.org/abs/2205.13147


Q2. 
What are the benefits of using an `async` approach to collecting our embeddings?
> NOTE: Determining the core difference between `async` and `sync` will be useful! If you get stuck - ask ChatGPT!

A2. 
The key benefit of using async is to improve performance by allowing processes to execute concurrently (asynchronously) rather than sequentially (synchronously). There is no need for sequential processing of collecting embeddings because these are independent lookup operations. This therefore allows for massive scalability, enabling the program to handle orders of magnitude more embedding requests.


Q3. 
When calling the OpenAI API - are there any ways we can achieve more reproducible outputs?
> NOTE: Check out [this section](https://platform.openai.com/docs/guides/text-generation/) of the OpenAI documentation for the answer!

A3.


#### ❓ Question #4:

What prompting strategies could you use to make the LLM have a more thoughtful, detailed response?

What is that strategy called?

> NOTE: You can look through ["Accessing GPT-3.5-turbo Like a Developer"](https://colab.research.google.com/drive/1mOzbgf4a2SP5qQj33ZxTz2a01-5eXqk2?usp=sharing) for an answer to this question if you get stuck!
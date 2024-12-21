# Preface

```{epigraph}
Models tell you merely what something is like, not what something is.

-- Emanuel Derman
```


An alternative title of this book could have been "Language Models Behaving Badly". If you are coming from a background in financial modeling, you may have noticed the parallel with Emanuel Derman's seminal work "Models.Behaving.Badly" {cite}`derman2011models`. This parallel is not coincidental. Just as Derman cautioned against treating financial models as perfect representations of reality, this book aims to highlight the limitations and pitfalls of Large Language Models (LLMs) in practical applications.

The book "Models.Behaving.Badly" by Emanuel Derman, a former physicist and Goldman Sachs quant, explores how financial and scientific models can fail when we mistake them for reality rather than treating them as approximations full of assumptions.
The core premise of his work is that while models can be useful tools for understanding aspects of the world, they inherently involve simplification and assumptions. Derman argues that many financial crises, including the 2008 crash, occurred partly because people put too much faith in mathematical models without recognizing their limitations.

Like financial models that failed to capture the complexity of human behavior and market dynamics, LLMs have inherent constraints. They can hallucinate facts, struggle with logical reasoning, and fail to maintain consistency across long outputs. Their responses, while often convincing, are probabilistic approximations based on training data rather than true understanding even though humans insist on treating them as "machines that can reason".

Today, there is this growing pervasive belief that these models could solve any problem, understand any context, or generate any content as wished by the user. Moreover, language models that were initially designed to be next-token prediction machines and chatbots are now been twisted and wrapped into "reasoning" machines for further integration into technology products and daily-life workflows that control, affect, or decide daily actions of our lives. This technological optimism coupled with lack of understanding of the models' limitations may pose risks we are still trying to figure out.

This book serves as an introductory, practical guide for practitioners and technology product builders - software engineers, data scientists, and product managers - who want to create the next generation of GenAI-based products with LLMs while remaining clear-eyed about their limitations and therefore their implications to end-users. Through detailed technical analysis, reproducible Python code examples we explore the gap between LLM capabilities and reliable software product development.

The goal is not to diminish the transformative potential of LLMs, but rather to promote a more nuanced understanding of their behavior. By acknowledging and working within their constraints, developers can create more reliable and trustworthy applications. After all, as Derman taught us, the first step to using a model effectively is understanding where it breaks down.

```{bibliography}
:filter: docname in docnames
```
# Markdown Files

Whether you write your book's content in Jupyter Notebooks (`.ipynb`) or
in regular markdown files (`.md`), you'll write in the same flavor of markdown
called **MyST Markdown**.
This is a simple file to help you get started and show off some syntax.

## What is MyST?

MyST stands for "Markedly Structured Text". It
is a slight variation on a flavor of markdown called "CommonMark" markdown,
with small syntax extensions to allow you to write **roles** and **directives**
in the Sphinx ecosystem.

For more about MyST, see [the MyST Markdown Overview](https://jupyterbook.org/content/myst.html).

## Sample Roles and Directives

Roles and directives are two of the most powerful tools in Jupyter Book. They
are like functions, but written in a markup language. They both
serve a similar purpose, but **roles are written in one line**, whereas
**directives span many lines**. They both accept different kinds of inputs,
and what they do with those inputs depends on the specific role or directive
that is being called.

Here is a "note" directive:

```{note}
Here is a note
```

It will be rendered in a special box when you build your book.

Here is an inline directive to refer to a document: {doc}`markdown-notebooks`.


## Citations

You can also cite references that are stored in a `bibtex` file. For example,
the following syntax: `` {cite}`holdgraf_evidence_2014` `` will render like
this: {cite}`holdgraf_evidence_2014`.

Moreover, you can insert a bibliography into your page with this syntax:
The `{bibliography}` directive must be used for all the `{cite}` roles to
render properly.
For example, if the references for your book are stored in `references.bib`,
then the bibliography is inserted with:

```{bibliography}
```

## Learn more

This is just a simple starter to get you started.
You can learn a lot more at [jupyterbook.org](https://jupyterbook.org).


Hands-On Large Language Models is poised to become the main introductory reference to LLMs. The book does an excellent job of covering what LLMs are, key components, how they work, their architectures, their capabilities and key use cases. 

AI Engineering is an exceptionally written exceedignly detailed book that covers more practical aspects of building and deploying LLMs-based applications. It covers a wide range of topics from model evaluation to deployment, observability to cost optimization.
- Little to no code making it less practical for beginners


However, the books still presents a major gap covering the practical limitations of LLMs.

| LLM Limitation | Problem Statement | Hands-On LLMs |
|----------------|-------------|---------------|
| **Structural (un)Reliability** | LLMs struggle to maintain consistent output formats, complicating their integration into larger systems and making error handling more complex. | Little to no coverage |
| **Size and Length Constraints** | LLMs have strict token limits for both inputs and outputs, requiring careful chunking and management strategies to handle long-form content effectively. | No coverage for output token limits. Limited coverage for input token limits implicitly coverage as a use case of RAGs in Chapter 8 |
| **Testing Complexity** | Traditional software testing methodologies break down when dealing with non-deterministic and generative systems, requiring new approaches. | Very limited coverage. While the topic is pertinent to LLMs in general, in the book it is briefly mentioned in the chapter about "Fine-Tuning Generation Models" |
| **Hallucination Management** | These models can generate plausible-sounding but entirely fabricated information, creating significant risks for production applications. | Very limited coverage. In Chapter 8, managing hallucinations is mentioned as a potential use case for RAGs, but no further details are provided. |
| **Safety and Security** | LLMs can generate harmful, biased, or inappropriate content, requiring robust safeguards and monitoring systems to ensure safe deployment. | No coverage |
| **Cost Optimization** | The computational and financial costs of operating LLM-based systems can quickly become prohibitive without careful management, observability and optimization. | No coverage |
| **Vendor Lock-in** | Cloud-based LLM providers can create significant dependencies and lock-in through their proprietary APIs and infrastructure, making it difficult to switch providers or self-host solutions. | Limited coverage. The book has a bias towards proprietary cloud-based solutions and does not cover self-hosting or alternative providers. |



This Book: 
- Centred on practical challenges and pitfalls in LLM-based applications 
- Emphasis on reliability and robustness 
- Offers balanced examination of both capabilities and limitations, helping readers make informed decisions about LLM integration. 
Competition: 
- Broader coverage of LLM fundamentals and theory 
- More focus on technical implementation 
- Heavy emphasis on tutorials and frameworks 
- Coverage of model architectures 




Evaluating Generative Models inside Fine-Tuning Generation Models

One limitation of Transformer language models is that they are limited in context sizes inside Semantic Search and Retrieval-Augmented Generation
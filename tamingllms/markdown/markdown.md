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



I am Tharsis Souza, a computer scientist holding a Doctorate in Computer Science (Ph.D.) from University College London (UCL), University of London, specializing in Artificial Intelligence (AI)-driven systems and data-driven products. My academic background encompasses a Ph.D., Master of Philosophy (M.Phil.), Master of Science (M.Sc.) in Computer Science, and a Bachelor of Science (B.Sc.) in Computer Engineering. I have authored or co-authored more than 12 scholarly publications, garnering approximately 500 citations in the fields of Natural Language Processing (NLP), Financial Computing, and Applied Computer Science. Additionally, I am frequently invited as a speaker at academic and business conferences. 
Professionally, I possess experience in developing Technology and AI-powered products at a variety of companies from start-ups to Fortune 500’s in the U.S., Brazil, and the U.K. I have built AI-based products at both a Silicon Valley startup and in Wall Street. Recently, I held the position of Senior VP at Two Sigma Investments, where I played a pivotal role in fostering the adoption of Large Language Models (LLMs) across the organization. Currently, I serve as a Lecturer at Columbia University's Master of Science (MSc.) program in Applied Analytics, where I teach the course "Solving Real-World Problems with Analytics". I also provide advisory services to Code.org on the subject of GenAI and mentor underrepresented groups in the "LatinX in AI" program. Moreover, I am an advocate for Open Source Software (OSS) (e.g. see www.podcastfy.ai, an open source alternative to Google's NotebookLM, powered by GenAI). 
Most recently, I have accepted the position of Head of Product, Equities at Citadel. 



 
Unique characteristics or experiences: 
1. Balanced Perspective: 
○ Bridge between academic research and industry implementation 
○ Experience scaling systems from proof-of-concept to production 
○ Understanding of technical, product and business considerations 
2. Teaching and Communication: 
○ Active faculty member teaching applied analytics at the Graduate level ○ Regular conference speaker on AI and technology 
○ Experience mentoring diverse groups of students and professionals 3. Cross-Industry Experience:
○ Built systems across different scales (startup to enterprise) 
○ Worked in multiple countries and technology ecosystems 
○ Direct experience with the challenges this book addresses 
This combination of academic depth, practical implementation experience, and teaching ability uniquely positions me to guide readers through the complexities of building reliable LLM-powered systems. My background enables me to present both theoretical foundations and practical solutions in an accessible, actionable way. 


This book focuses on the practical challenges and solutions in implementing Large Language Models (LLMs)-powered products. While many resources cover the capabilities of LLMs, this book specifically addresses the hidden complexities and pitfalls that engineers and technical product managers face when building reliable LLM-powered applications. 
In recent years, Large Language Models (LLMs) have emerged as a transformative force in technology, promising to revolutionize how we build products and interact with computers. From ChatGPT to GitHub Copilot, Claude Artifacts, cursor.com, replit, and others, these systems have captured the public imagination and sparked a gold rush of AI-powered applications. However, beneath the surface of this technological revolution lies a complex landscape of challenges that practitioners must navigate. 
As we’ll explore in this book, the engineering effort required to manage these challenges - from handling non-deterministic outputs to preventing hallucinations - cannot be overstated. While the potential of LLM technology remains compelling, understanding and addressing the hidden costs and complexities of building reliable LLM-powered systems will enable us to fully harness their transformative impact. 

Large Language Models (LLMs) promise to revolutionize how we build software, but implementing them reliably remains challenging. In "Taming Language Models with Open Source," Dr. Tharsis Souza provides a practical guide to overcoming LLM limitations using open source solutions.



Dr. Tharsis Souza is a computer scientist and product leader specializing in AI-based product development. He is a Lecturer at Columbia University's Master of Science program in Applied Analytics, Head of Product, Equities at Citadel, and former Senior VP at Two Sigma Investments. 

With over 15 years of experience delivering technology products across startups and Fortune 500 companies globally, Dr. Souza is also an author of numerous scholarly publications and is a frequent speaker at academic and business conferences. Grounded on academic background and drawing from practical experience building and scaling up language models-based products at major institutions, early-stage startups as well as advising non-profit organizations, and contributing to open source projects, he brings a unique perspective on bridging the gap between LLMs potential and their practical limitations using open source tools to enable the next generation of AI-powered products.

Dr. Tharsis holds a Ph.D. in Computer Science from UCL, University of London following an M.Phil. and M.Sc. in Computer Science and a B.Sc. in Computer Engineering.
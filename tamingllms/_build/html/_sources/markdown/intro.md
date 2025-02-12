---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(intro)=
# About the Book

```{epigraph}
I am always doing that which I cannot do, in order that I may learn how to do it.

-- Pablo Picasso
```
```{contents}
```

## Core Challenges We'll Address

In recent years, Large Language Models (LLMs) have emerged as a transformative force in technology, promising to revolutionize how we build products and interact with computers. From ChatGPT and LLama to GitHub Copilot and Claude Artifacts these systems have captured the public imagination and sparked a gold rush of AI-powered applications. However, beneath the surface of this technological revolution lies a complex landscape of challenges that software developers and tech leaders must navigate. 

This book focuses on bringing awareness to key LLM limitations and harnessing open source solutions to overcome them for building robust AI-powered products. It offers a critical perspective on implementation challenges, backed by practical and reproducible Python examples. While many resources cover the capabilities of LLMs, this book specifically addresses the hidden complexities and pitfalls that engineers and technical leaders face when building LLM-powered applications while offering a comprehensive guide on how to leverage battle-tested open source tools and solutions.


Throughout this book, we'll tackle the following (non-exhaustive) list of critical challenges:

1. **Structural (un)Reliability**: LLMs struggle to maintain consistent output formats, complicating their integration into larger systems and making error handling more complex.

2. **Input Data Management**: LLMs are sensitive to input data format, operate with stale data and struggle with long-context requiring careful input data management and retrieval strategies.

3. **Testing Complexity**: Traditional software testing methodologies break down when dealing with non-deterministic and generative systems, requiring new approaches.

4. **Safety**: LLMs can generate harmful, biased, or inappropriate content, requiring robust safeguards and monitoring systems to ensure safe deployment.

5. **Alignment**: LLMs are next-token prediction models, which means they are not aligned with the user's preferences by default.

6. **Vendor Lock-in**: Cloud-based LLM providers can create significant dependencies and lock-in through their proprietary APIs and infrastructure, making it difficult to switch providers or self-host solutions.

7. **Cost Optimization**: The computational and financial costs of operating LLM-based systems can quickly become prohibitive without careful management, and optimization.

We conclude with a discussion on the future of LLMs and the challenges that will arise as we move forward.


## A Practical Approach

This book takes a hands-on approach to these challenges, with a focus on accessibility and reproducibility. 
All examples and code are:

- Fully reproducible and documented, allowing readers to replicate results exactly
- Designed to run on consumer-grade hardware without requiring expensive resources
- Available as open source Python notebooks that can be modified and extended
- Structured to minimize computational costs while maintaining effectiveness

## An Open Source Approach

Throughout this book, we'll leverage open source tools and frameworks to address common LLM challenges. In that way, we are prioritizing:

- **Transparency**: Open source solutions provide visibility into how challenges are being addressed, allowing for better understanding and customization of solutions.
- **Flexibility**: Open source tools can be modified and adapted to specific use cases, unlike black-box commercial solutions.
- **Cost-Effectiveness**: Most of open source tools we will cover are freely available, fostering accessibility and reducing costs.
- **Vendor Independence**: Open source solutions reduce dependency on specific providers, offering more freedom in architectural decisions.

## Open Source Book

In keeping with these open source principles, this book itself is open source and available on GitHub. It's designed to be a living document that evolves with the changing landscape of LLM technology and implementation practices. Readers are encouraged to:

- Report issues or suggest improvements through GitHub Issues
- Contribute new examples or solutions via Pull Requests
- Share their own experiences and solutions with the community
- Propose new chapters or sections that address emerging challenges

The repository can be found at https://github.com/souzatharsis/tamingllms. Whether you've found a typo, have a better solution to share, or want to contribute, your contributions are welcome. Please feel free to open an issue in the book repository.


## A Note on Perspective

While this book takes a critical look at LLM limitations, our goal is not to discourage their use but to enable more robust and reliable implementations. By understanding these challenges upfront, you'll be better equipped to build systems that leverage LLMs effectively while avoiding common pitfalls.

The current discourse around LLMs tends toward extremes - either uncritical enthusiasm or wholesale dismissal. This book takes a different approach:

- **Practical Implementation Focus**: Rather than theoretical capabilities, we examine practical challenges and their solutions.
- **Code-First Learning**: Every concept is illustrated with executable Python examples, enabling immediate practical application.
- **Critical Analysis**: We provide a balanced examination of both capabilities and limitations, helping readers make informed decisions about LLM integration.

## Who This Book Is For

This book is designed for: 

- Software/AI Engineers building LLM-powered applications 
- Technical Product Managers leading GenAI initiatives 
- Technical Leaders making architectural decisions
- Open Source advocates and/or developers building LLM Applications 
- Anyone seeking to understand the practical challenges of working with LLMs 

Typical job roles:

- Software/AI Engineers building AI-powered platforms
- Backend Developers integrating LLMs into existing systems
- ML Engineers transitioning to LLM implementation
- Technical Leads making architectural decisions
- Product Managers overseeing GenAI initiatives

Reader motivation:

- Need to build reliable, production-ready LLM applications
- Desire to understand and overcome common LLM implementation challenges
- Requirement to optimize costs and performance
- Need to ensure safety and reliability in LLM-powered systems

The goal is to help readers understand and address these challenges early, before they become costly problems too late in the software development lifecycle. 

## Outcomes

After reading this book, the reader will understand critical LLM limitations and their implications and have practical experience on recommended open source tools and frameworks to help navigate common LLM pitfalls. The reader will be able to:

- Implement effective strategies for managing LLMs limitations
- Build reliable LLM-powered applications
- Create robust testing frameworks for LLM-based systems
- Deploy proper LLM safeguards
- Make realistic effort estimations for LLM-based projects
- Understand the hidden complexities that impact development timelines

## Prerequisites

To make the most of this book, you should have:

- Basic Python programming experience
- Basic knowledge of LLMs and their capabilities
- Access to and basic knowledge of LLM APIs (Mistral, OpenAI, Anthropic, or similar)
- A desire to build reliable LLM-based applications

## Setting Up Your Environment

Before diving into the examples in this book, you'll need to set up your development environment. Here's how to get started:

### Code Repository
Clone the book's companion repository:
```bash
git clone https://github.com/souzatharsis/tamingllms.git
cd tamingllms/notebooks
```

### Python Environment Setup
```bash
# Create and activate a virtual environment
python -m venv taming-llms-env
source taming-llms-env/bin/activate  # On Windows, use: taming-llms-env\Scripts\activate
```
We will try and make each Chapter as self-contained as possible, including all necessary installs as we go through the examples.
Feel free to use your preferred package manager to install the dependencies (e.g. `pip`). We used `poetry` to manage dependencies and virtual environments.

### API Keys Configuration
1. Create a `.env` file in the root directory of the project.
2. Add your API keys and other sensitive information to the `.env` file. For example:

   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

```{note}
Never share your `.env` file or commit it to version control. It contains sensitive information that should be kept private.
```

### Troubleshooting Common Issues
- If you encounter API rate limits, consider using smaller examples or implementing retry logic
- For package conflicts, try creating a fresh virtual environment or use a package manager like `poetry`
- Check the book's repository issues page for known problems and solutions

Now that your environment is set up, let's begin our exploration of LLM challenges.

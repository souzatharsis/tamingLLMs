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
# Introduction

```{epigraph}
I am always doing that which I cannot do, in order that I may learn how to do it.

-- Pablo Picasso
```
```{contents}
```

## Core Challenges We'll Address
In recent years, Large Language Models (LLMs) have emerged as a transformative force in technology, promising to revolutionize how we build products and interact with computers. From ChatGPT to GitHub Copilot, Claude Artifacts, cursor.com, replit, and others, these systems have captured the public imagination and sparked a gold rush of AI-powered applications. However, beneath the surface of this technological revolution lies a complex landscape of challenges that practitioners must navigate. 

As we'll explore in this book, the engineering effort required to manage these challenges - from handling non-deterministic outputs to preventing hallucinations - cannot be overstated. While the potential of LLM technology remains compelling, understanding and addressing the hidden costs and complexities of building reliable LLM-powered systems will enable us to fully harness their transformative impact.

While the capabilities of LLMs are indeed remarkable, the prevailing narrative often glosses over fundamental problems that engineers, product managers, and organizations face when building real-world applications. This book aims to bridge that gap, offering a practical, clear-eyed examination of the pitfalls and challenges in working with LLMs.

Throughout this book, we'll tackle the following (non-exhaustive) list of critical challenges:

1. **Non-deterministic Behavior**: Unlike traditional software systems, LLMs can produce different outputs for identical inputs, making testing and reliability assurance particularly challenging.

2. **Structural (un)Reliability**: LLMs struggle to maintain consistent output formats, complicating their integration into larger systems and making error handling more complex.

3. **Hallucination Management**: These models can generate plausible-sounding but entirely fabricated information, creating significant risks for production applications.

4. **Cost Optimization**: The computational and financial costs of operating LLM-based systems can quickly become prohibitive without careful optimization.

5. **Testing Complexity**: Traditional testing methodologies break down when dealing with non-deterministic systems, requiring new approaches.

## A Practical Approach

This book takes a hands-on approach to these challenges, providing:

- Concrete Python examples that you can run and modify
- Real-world scenarios and solutions
- Testing strategies and best practices
- Cost optimization techniques
- Integration patterns and anti-patterns

## A Note on Perspective

While this book takes a critical look at LLM limitations, our goal is not to discourage their use but to enable more robust and reliable implementations. By understanding these challenges upfront, you'll be better equipped to build systems that leverage LLMs effectively while avoiding common pitfalls.

The current discourse around LLMs tends toward extremes—either uncritical enthusiasm or wholesale dismissal. This book takes a different approach:

- **Practical Implementation Focus**: Rather than theoretical capabilities, we examine real-world challenges and their solutions.
- **Code-First Learning**: Every concept is illustrated with executable Python examples, enabling immediate practical application.
- **Critical Analysis**: We provide a balanced examination of both capabilities and limitations, helping readers make informed decisions about LLM integration.

## Who This Book Is For

This book is designed for:

- Software Engineers building LLM-powered applications
- Product Managers leading AI initiatives
- Technical Leaders making architectural decisions
- Anyone seeking to understand the practical challenges of working with LLMs

Typical job roles:

- Software Engineers building AI-powered platforms
- Backend Developers integrating LLMs into existing systems
- ML Engineers transitioning to LLM implementation
- Technical Leads making architectural decisions
- Product Managers overseeing GenAI initiatives

Reader motivation:

- Need to build reliable, production-ready LLM applications
- Desire to understand and overcome common LLM implementation challenges
- Requirement to optimize costs and performance
- Need to ensure safety and reliability in LLM-powered systems

## Outcomes

After reading this book, the reader will be able to:
- Build reliable LLM-powered applications
- Implement effective strategies for managing LLMs limitations
- Create robust testing frameworks for LLM-based systems
- Deploy proper LLM safeguards
- Make realistic effort estimations for LLM-based projects
- Understand the hidden complexities that impact development timelines

## Prerequisites

To make the most of this book, you should have:

- Basic Python programming experience
- Access to and basic knowledge of LLM APIs (OpenAI, Anthropic, or similar)
- A desire to build reliable, production-grade LLM-powered products


## Setting Up Your Environment

Before diving into the examples in this book, you'll need to set up your development environment. Here's how to get started:

### Python Environment Setup
```bash
# Create and activate a virtual environment
python -m venv llm-book-env
source llm-book-env/bin/activate  # On Windows, use: llm-book-env\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

### API Keys Configuration
1. Create a `.env` file in the root directory of the project.
2. Add your API keys and other sensitive information to the `.env` file. For example:

   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ```

```{note}
Never share your `.env` file or commit it to version control. It contains sensitive information that should be kept private.
```

### Code Repository
Clone the book's companion repository:
```bash
git clone https://github.com/souzatharsis/tamingllms.git
cd tamingllms
```

### Troubleshooting Common Issues
- If you encounter API rate limits, consider using smaller examples or implementing retry logic
- For package conflicts, try creating a fresh virtual environment or use a package manager like `poetry`
- Check the book's repository issues page for known problems and solutions

Now that your environment is set up, let's begin our exploration of LLM challenges.

## About the Author(s)

Dr. Tharsis Souza is a computer scientist and product leader specializing in AI-based product development. He is a Lecturer at Columbia University's Master of Science program in Applied Analytics, Head of Product, Equities at Citadel, and former Senior VP at Two Sigma Investments. 

With over 15 years of experience delivering technology products across startups and Fortune 500 companies globally, Dr. Souza is also an author of numerous scholarly publications and is a frequent speaker at academic and business conferences. Grounded on academic background and drawing from practical experience building and scaling up products powered by language models at early-stage startups, major institutions as well as advising non-profit organizations, and contributing to open source projects, he brings a unique perspective on bridging the gap between LLMs promised potential and their practical limitations using open source tools to enable the next generation of AI-powered products.

Dr. Tharsis holds a Ph.D. in Computer Science from UCL, University of London following an M.Phil. and M.Sc. in Computer Science and a B.Sc. in Computer Engineering.
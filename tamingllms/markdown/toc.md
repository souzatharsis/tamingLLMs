---
title: "Taming LLMs: A Practical Guide to LLM Pitfalls with Open Source Software"
author: "Tharsis T. P. Souza"
date: "2024-11-22"
---

# Taming LLMs
## *A Practical Guide to LLM Pitfalls with Open Source Software*


Abstract: *The current discourse around Large Language Models (LLMs) tends to focus heavily on their capabilities while glossing over fundamental challenges. Conversely, this book takes a critical look at the key limitations and implementation pitfalls that engineers and technical product managers encounter when building LLM-powered applications. Through practical Python examples and proven open source solutions, it provides an introductory yet comprehensive guide for navigating these challenges. The focus is on concrete problems - from handling unstructured output to managing context windows - with reproducible code examples and battle-tested open source tools. By understanding these pitfalls upfront, readers will be better equipped to build products that harness the power of LLMs while sidestepping their inherent limitations.*


## Chapter 1: Introduction
- 1.1 Core Challenges We'll Address
- 1.2 A Practical Approach
- 1.3 A Note on Perspective
- 1.4 Who This Book Is For
- 1.5 Outcomes
- 1.6 Prerequisites
- 1.7 Setting Up Your Environment
  - 1.7.1 Python Environment Setup
  - 1.7.2 API Keys Configuration
  - 1.7.3 Code Repository
  - 1.7.4 Troubleshooting Common Issues
- 1.8 About the Author(s)

## Chapter 2: Wrestling with Structured Output
- 2.1 The Structured Output Challenge
- 2.2 Problem Statement
- 2.3 Solutions
  - 2.3.1 Strategies
  - 2.3.2 Techniques and Tools
    - 2.3.2.1 One-Shot Prompts
    - 2.3.2.2 Structured Output with Provider-Specific APIs
      - 2.3.2.2.1 JSON Mode
      - 2.3.2.2.2 Structured Output Mode
    - 2.3.2.3 LangChain
    - 2.3.2.4 Outlines
      - 2.3.2.4.1 Multiple Choice Generation
      - 2.3.2.4.2 Pydantic model
- 2.4 Discussion
  - 2.4.1 Comparing Solutions
  - 2.4.2 Best Practices
  - 2.4.3 Research & Ongoing Debate
- 2.5 Conclusion
- 2.6 Acknowledgements
- 2.7 References

## Chapter 3: Input Size and Length Limitations
- 3.1 Context Window Constraints
- 3.2 Handling Long Inputs
- 3.3 Managing Token Limits
- 3.4 Chunking Strategies
- 3.5 Implementation Patterns
- 3.6 Testing Long-form Content

## Chapter 4: Output Size and Length Limitations
- 4.1 What are Token Limits?
- 4.2 Problem Statement
- 4.3 Content Chunking with Contextual Linking
  - 4.3.1 Generating long-form content
  - 4.3.2 Step 1: Chunking the Content
  - 4.3.3 Step 2: Writing the Base Prompt Template
  - 4.3.4 Step 3: Constructing Dynamic Prompt Parameters
  - 4.3.5 Step 4: Generating the Report
  - 4.3.6 Example Usage
- 4.4 Discussion
- 4.5 Implications
- 4.6 Future Considerations
- 4.7 Conclusion
- 4.8 References

## Chapter 5: The Evals Gap
- 5.1 Non-Deterministic Machines
  - 5.1.1 Temperature and Sampling
  - 5.1.2 The Temperature Spectrum
- 5.2 Emerging Properties
- 5.3 Problem Statement
- 5.4 Evals Design
  - 5.4.1 Conceptual Overview
  - 5.4.2 Design Considerations
  - 5.4.3 Key Components
  - 5.4.4 Metrics
    - 5.4.4.1 Working Example
    - 5.4.4.2 Considerations
  - 5.4.5 Evaluators
    - 5.4.5.1 Model-Based Evaluation
    - 5.4.5.2 Human-Based Evaluation
  - 5.4.6 Benchmarks & Leaderboards
  - 5.4.7 Tools
- 5.5 References

## Chapter 6: Hallucination: The Reality Gap
- 6.1 Understanding Hallucination Types
- 6.2 Detection Strategies
- 6.3 Grounding Techniques
- 6.4 Retrieval-Augmented Generation (RAG)
  - 6.4.1 Context Selection
  - 6.4.2 Indexing Strategies
  - 6.4.3 Vector Stores
  - 6.4.4 Chunking Methods
- 6.5 Practical Implementation
  - 6.5.1 Building a RAG Pipeline
  - 6.5.2 Testing and Validation

## Chapter 7: Safety Concerns
- 7.1 Common Safety Issues
- 7.2 Implementation of Safety Guards
- 7.3 Content Filtering
- 7.4 Input Validation
- 7.5 Output Sanitization
- 7.6 Monitoring and Alerts
- 7.7 Best Practices

## Chapter 8: The Cost Factor
- 8.1 Understanding LLM Costs
- 8.2 Token Optimization
- 8.3 Caching Strategies
  - 8.3.1 Implementation Patterns
  - 8.3.2 Cache Invalidation
- 8.4 Output Prediction Techniques
- 8.5 Cost Monitoring
- 8.6 Optimization Strategies

## Chapter 9: Breaking Free from Cloud Providers
- 9.1 The Vendor Lock-in Problem
- 9.2 Self-hosting Solutions
  - 9.2.1 Llama Implementation
  - 9.2.2 Llamafile Setup and Usage
  - 9.2.3 Ollama Deployment
- 9.3 Performance Considerations
- 9.4 Cost Analysis
- 9.5 Migration Strategies


## Appendix A: Tools and Resources
- A.1 Evaluation Tools
- A.2 Monitoring Solutions
- A.3 Open Source Models
- A.4 Community Resources


## Citation
[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC-BY--NC--SA-4.0-lightgrey.svg

```
@misc{tharsistpsouza2024tamingllms,
  author = {Tharsis T. P. Souza},
  title = {Taming LLMs: A Practical Guide to LLM Pitfalls with Open Source Software},
  year = {2024},
  journal = {GitHub repository},
  url = {https://github.com/souzatharsis/tamingLLMs)
}
```
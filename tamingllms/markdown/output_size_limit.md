# Output Size Limitations
```{epigraph}
Only those who will risk going too far can possibly find out how far one can go.

-- T.S. Eliot
```
```{contents}
:depth: 2
```
## What are Token Limits?

Tokens are the basic units that LLMs process text with. A token can be as short as a single character or as long as a complete word. In English, a general rule of thumb is that 1 token ≈ 4 characters or ¾ of a word.

The `max_output_tokens` is parameter often available in modern LLMs that determines the maximum length of text that an LLM can generate in a single response. {numref}`token-cost-table` shows the `max_output_tokens` for several key models, which typically range between 4096 and 16384 tokens. Contrary to what one might expect, the model does not "summarizes the answer" such that it does not surpass `max_output_tokens` limit. Instead, it will stop once it reaches this limit, even mid-sentence, i.e. the response may be truncated.

```{table} Token Cost and Length Limitation Comparison Across Key Models
:name: token-cost-table
| Model                        | max_output_tokens | max_input_tokens | input_cost_per_token | output_cost_per_token |
|------------------------------|-------------------|------------------|----------------------|-----------------------|
| meta.llama3-2-11b-instruct-v1:0 | 4096              | 128000           | 3.5e-7               | 3.5e-7                |
| claude-3-5-sonnet-20241022   | 8192              | 200000           | 3e-6                 | 1.5e-5                |
| gpt-4-0613                   | 4096              | 8192             | 3e-5                 | 6e-5                  |
| gpt-4-turbo-2024-04-09       | 4096              | 128000           | 1e-5                 | 3e-5                  |
| gpt-4o-mini                  | 16384             | 128000           | 1.5e-7               | 6e-7                  |
| gemini/gemini-1.5-flash-002  | 8192              | 1048576          | 7.5e-8               | 3e-7                  |
| gemini/gemini-1.5-pro-002    | 8192              | 2097152          | 3.5e-6               | 1.05e-5               |
```

## Problem Statement

The `max_output_tokens` limit in LLMs poses a significant challenge for users who need to generate long outputs, as it may result in truncated content and/or incomplete information.

1. **Truncated Content**: Users aiming to generate extensive content, such as detailed reports or comprehensive articles, may find their outputs abruptly cut off due to the `max_output_tokens` limit. This truncation can result in incomplete information and disrupt the flow of the content.

2. **Shallow Responses**: When users expect a complete and thorough response but receive only a partial output, it can lead to dissatisfaction and frustration. This is especially true in applications where the completeness of information is critical, such as in educational tools or content creation platforms.

To effectively address these challenges, developers need to implement robust solutions that balance user expectations with technical and cost constraints, ensuring that long-form content generation remains feasible and efficient.

## Content Chunking with Contextual Linking

Content chunking with contextual linking is a technique used to manage the `max_output_tokens` limitation by breaking down long-form content into smaller, manageable chunks. This approach allows the LLM to focus on smaller sections of the input, enabling it to generate more complete and detailed responses for each chunk while maintaining coherence and context across the entire output.

1. **Chunking the Content**: The input content is split into smaller chunks. This allows the LLM to process each chunk individually, focusing on generating a complete and detailed response for that specific section of the input.

2. **Maintaining Context**: Each chunk is linked with contextual information from the previous chunks. This helps in maintaining the flow and coherence of the content across multiple chunks.

3. **Generating Linked Prompts**: For each chunk, a prompt is generated that includes the chunk's content and its context. This prompt is then used to generate the output for that chunk.

4. **Combining the Outputs**: The outputs of all chunks are combined to form the final long-form content.

By following these steps, developers can effectively manage the `max_output_tokens` limitation and generate coherent long-form content without truncation.

Let's examine an example implementation of this technique.

### Generating long-form content

- Goal: Generate a long-form report analyzing a company's financial statement.
- Input: A company's 10K SEC filing.

![Content Chunking with Contextual Linking](../_static/structured_output/diagram1.png)


#### Step 1: Chunking the Content

There are different methods for chunking, and each of them might be appropriate for different situations. However, we can broadly group chunking strategies in two types:
- **Fixed-size Chunking**: This is the most common and straightforward approach to chunking. We simply decide the number of tokens in our chunk and, optionally, whether there should be any overlap between them. In general, we will want to keep some overlap between chunks to make sure that the semantic context doesn’t get lost between chunks. Fixed-sized chunking may be a reasonable path in many common cases. Compared to other forms of chunking, fixed-sized chunking is computationally cheap and simple to use since it doesn’t require the use of any specialied techniques or libraries.
- **Content-aware Chunking**: These are a set of methods for taking advantage of the nature of the content we’re chunking and applying more sophisticated chunking to it. Examples include:
  - **Sentence Splitting**: Many models are optimized for embedding sentence-level content. Naturally, we would use sentence chunking, and there are several approaches and tools available to do this, including naive splitting (e.g. splitting on periods), NLTK, and spaCy.
  - **Recursive Chunking**: Recursive chunking divides the input text into smaller chunks in a hierarchical and iterative manner using a set of separators.
  - **Semantic Chunking**: This is a class of methods that leverages embeddings to extract the semantic meaning present in your data, creating chunks that are made up of sentences that talk about the same theme or topic.

Here, we will utilize `langchain` for a content-aware sentence-splitting strategy for chunking. We will use the `CharacterTextSplitter` with `tiktoken` as our tokenizer to count the number of tokens per chunk which we can use to ensure that we do not surpass the input token limit of our model.


```python
def get_chunks(text: str, chunk_size: int, chunk_overlap: int) -> list:
    """
    Split input text into chunks of specified size with specified overlap.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The maximum size of each chunk in tokens.
        chunk_overlap (int): The number of tokens to overlap between chunks.

    Returns:
        list: A list of text chunks.
    """
    from langchain_text_splitters import CharacterTextSplitter

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_text(text)

```

We will write a base prompt template which will serve as a foundational structure for all chunks, ensuring consistency in the instructions and context provided to the language model. The template includes the following parameters:
- `role`: Defines the role or persona the model should assume.
- `context`: Provides the background information or context for the task.
- `instruction`: Specifies the task or action the model needs to perform.
- `input_text`: Contains the actual text input that the model will process.
- `requirements`: Lists any specific requirements or constraints for the output.


```python
from langchain_core.prompts import PromptTemplate
def get_base_prompt_template() -> str:
    
    base_prompt = """
    ROLE: {role}
    CONTEXT: {context}
    INSTRUCTION: {instruction}
    INPUT: {input}
    REQUIREMENTS: {requirements}
    """
    
    prompt = PromptTemplate.from_template(base_prompt)
    return prompt
```

We will write a simple function that returns an `LLMChain` which is a simple `langchain` construct that allows you to chain together a combination of prompt templates, language models and output parsers.


```python
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatLiteLLM

def get_llm_chain(prompt_template: str, model_name: str, temperature: float = 0):
    """
    Returns an LLMChain instance using langchain.

    Args:
        prompt_template (str): The prompt template to use.
        model_name (str): The name of the model to use.
        temperature (float): The temperature setting for the model.

    Returns:
        llm_chain: An instance of the LLMChain.
    """
    
    from dotenv import load_dotenv
    import os

    # Load environment variables from .env file
    load_dotenv()
    
    api_key_label = model_name.split("/")[0].upper() + "_API_KEY"
    llm = ChatLiteLLM(
        model=model_name,
        temperature=temperature,
        api_key=os.environ[api_key_label],
    )
    llm_chain = prompt_template | llm | StrOutputParser()
    return llm_chain
```

Now, we will write a function (`get_dynamic_prompt_template`) that constructs prompt parameters dynamically for each chunk.


```python
from typing import Dict
def get_dynamic_prompt_params(prompt_params: Dict, 
                            part_idx: int, 
                            total_parts: int,
                            chat_context: str,
                            chunk: str) -> str:
    """
    Construct prompt template dynamically per chunk while maintaining the chat context of the response generation.
    
    Args:
        prompt_params (Dict): Original prompt parameters
        part_idx (int): Index of current conversation part
        total_parts (int): Total number of conversation parts
        chat_context (str): Chat context from previous parts
        chunk (str): Current chunk of text to be processed
    Returns:
        str: Dynamically constructed prompt template with part-specific params
    """
    dynamic_prompt_params = prompt_params.copy()
    # saves the chat context from previous parts
    dynamic_prompt_params["context"] = chat_context
    # saves the current chunk of text to be processed as input
    dynamic_prompt_params["input"] = chunk
    
    # Add part-specific instructions
    if part_idx == 0: # Introduction part
        dynamic_prompt_params["instruction"] = f"""
        You are generating the Introduction part of a long report.
        Don't cover any topics yet, just define the scope of the report.
        """
    elif part_idx == total_parts - 1: # Conclusion part
        dynamic_prompt_params["instruction"] = f"""
        You are generating the last part of a long report. 
        For this part, first discuss the below INPUT. Second, write a "Conclusion" section summarizing the main points discussed given in CONTEXT.
        """
    else: # Main analysis part
        dynamic_prompt_params["instruction"] = f"""
        You are generating part {part_idx+1} of {total_parts} parts of a long report.
        For this part, analyze the below INPUT.
        Organize your response in a way that is easy to read and understand either by creating new or merging with previously created structured sections given in CONTEXT.
        """
    
    return dynamic_prompt_params
```

Finally, we will write a function that generates the actual report by calling the `LLMChain` with the dynamically updated prompt parameters for each chunk and concatenating the results at the end.


```python
def generate_report(input_content: str, llm_model_name: str, 
                    role: str, requirements: str,
                    chunk_size: int, chunk_overlap: int) -> str:
    # stores the parts of the report, each generated by an individual LLM call
    report_parts = [] 
    # split the input content into chunks
    chunks = get_chunks(input_content, chunk_size, chunk_overlap)
    # initialize the chat context with the input content
    chat_context = input_content
    # number of parts to be generated
    num_parts = len(chunks)

    prompt_params = {
        "role": role, # user-provided
        "context": "", # dinamically updated per part
        "instruction": "", # dynamically updated per part
        "input": "", # dynamically updated per part
        "requirements": requirements #user-priovided
    }

    # get the LLMChain with the base prompt template
    llm_chain = get_llm_chain(get_base_prompt_template(), 
                                 llm_model_name)

    # dynamically update prompt_params per part
    print(f"Generating {num_parts} report parts")
    for i, chunk in enumerate(chunks):
        dynamic_prompt_params = get_dynamic_prompt_params(
            prompt_params,
            part_idx=i,
            total_parts=num_parts,
            chat_context=chat_context,
            chunk=chunk
        )
        
        # invoke the LLMChain with the dynamically updated prompt parameters
        response = llm_chain.invoke(dynamic_prompt_params)

        # update the chat context with the cummulative response
        if i == 0:
            chat_context = response
        else:
            chat_context = chat_context + response
            
        print(f"Generated part {i+1}/{num_parts}.")
        report_parts.append(response)

    report = "\n".join(report_parts)
    return report
```

## Example Usage


```python
# Load the text from sample 10K SEC filing
with open('../data/apple.txt', 'r') as file:
    text = file.read()
```


```python
# Define the chunk and chunk overlap size
MAX_CHUNK_SIZE = 10000
MAX_CHUNK_OVERLAP = 0
```


```python
report = generate_report(text, llm_model_name="gemini/gemini-1.5-flash-latest", 
                           role="Financial Analyst", 
                           requirements="The report should be in a readable, structured format, easy to understand and follow. Focus on finding risk factors and market moving insights.",
                           chunk_size=MAX_CHUNK_SIZE, 
                           chunk_overlap=MAX_CHUNK_OVERLAP)
```

    Generating 5 report parts
    Generated part 1/5.
    Generated part 2/5.
    Generated part 3/5.
    Generated part 4/5.
    Generated part 5/5.



```python
# Save the generated report to a local file
with open('data/apple_report.txt', 'w') as file:
    file.write(report)

```


```python
print(report[:300] + "\n...\n" + report[-300:])
```

    **Introduction**
    
    This report provides a comprehensive analysis of Apple Inc.'s financial performance and position for the fiscal year ended September 28, 2024, as disclosed in its Form 10-K filing with the United States Securities and Exchange Commission.  The analysis will focus on identifying key
    ...
    luation.  The significant short-term obligations, while manageable given Apple's cash position, highlight the need for continued financial discipline and effective risk management.  A deeper, more granular analysis of the financial statements and notes is recommended for a more complete assessment.
    


### Discussion

Results from the generated report present a few interesting aspects:

- **Coherence**: The generated report demonstrates a high level of coherence. The sections are logically structured, and the flow of information is smooth. Each part of the report builds upon the previous sections, providing a comprehensive analysis of Apple Inc.'s financial performance and key risk factors. The use of headings and subheadings helps in maintaining clarity and organization throughout the document.

- **Adherence to Instructions**: The LLM followed the provided instructions effectively. The report is in a readable, structured format, and it focuses on identifying risk factors and market-moving insights as requested. The analysis is detailed and covers various aspects of Apple's financial performance, including revenue segmentation, profitability, liquidity, and capital resources. The inclusion of market-moving insights adds value to the report, aligning with the specified requirements.

Despite the high quality of the results, there are some limitations to consider:

- **Depth of Analysis**: While the report covers a wide range of topics, the depth of analysis in certain sections may not be as comprehensive as a human expert's evaluation. Some nuances and contextual factors might be overlooked by the LLM. Splitting the report into multiple parts helps in mitigating this issue.

- **Chunking Strategy**: The current approach splits the text into chunks based on size, which ensures that each chunk fits within the model's token limit. However, this method may disrupt the logical flow of the document, as sections of interest might be split across multiple chunks. An alternative approach could be "structured" chunking, where the text is divided based on meaningful sections or topics. This would preserve the coherence of each section, making it easier to follow and understand. Implementing structured chunking requires additional preprocessing to identify and segment the text appropriately, but it can significantly enhance the readability and logical flow of the generated report.


## Implications

Implementing context chunking with contextual linking is a practical solution to manage the output size limitations of LLMs. However, this approach comes with its own set of implications that developers must consider.

1. **Increased Development Complexity**: Implementing strategies to overcome the maximum output token length introduces additional layers of complexity to the application design. It necessitates meticulous management of context across multiple outputs to maintain coherence. Ensuring that each chunk retains the necessary context for the conversation or document can be challenging and often requires advanced logic to handle transitions seamlessly.

2. **Cost Implications**: Attempting to circumvent the `max_output_tokens` limitation by making multiple requests can increase the number of tokens processed, thereby raising the operational costs associated with using LLM services. Each additional request contributes to the overall token usage, which can quickly escalate costs, especially for applications with high-frequency interactions or large volumes of data.

3. **Performance Bottlenecks**: Generating long outputs in segments can lead to performance bottlenecks, as each segment may require additional processing time and resources, impacting the overall efficiency of the application. The need to manage and link multiple chunks can introduce latency and reduce the responsiveness of the system, which is critical for real-time applications.

By understanding these implications, developers can better prepare for the challenges associated with context chunking and contextual linking, ensuring that their applications remain efficient, cost-effective, and user-friendly.


## Future Considerations

As models evolve, we can expect several advancements that will significantly impact how we handle output size limitations:

1. **Contextual Awareness**: Future LLMs will likely have improved contextual awareness - or as Mustafa Suleyman would call "infinite memory", enabling them to better understand and manage the context of a conversation or document over long interactions. This will reduce the need for repetitive context setting and improve the overall user experience.

2. **More Efficient Token Usage**: Advances in model architecture and tokenization strategies will lead to more efficient token usage. This means that models will be able to convey the same amount of information using fewer tokens, reducing costs and improving performance.

3. **Improved Compression Techniques**: As research progresses, we can expect the development of more sophisticated compression techniques that allow models to retain essential information while reducing the number of tokens required. This will be particularly useful for applications that need to summarize or condense large amounts of data.

4. **Adaptive Token Limits**: Future models may implement adaptive token limits that dynamically adjust based on the complexity and requirements of the task at hand. This will provide more flexibility and efficiency in handling diverse use cases.

5. **Enhanced Memory Management**: Innovations in memory management will allow models to handle larger outputs without a significant increase in computational resources. This will make it feasible to deploy advanced LLMs in resource-constrained environments.

These advancements will collectively enhance the capabilities of LLMs, making them more powerful and versatile tools for a wide range of applications. However, they will also introduce new challenges and considerations that developers and researchers will need to address to fully harness their potential.


## Conclusion

In conclusion, while managing output size limitations in LLMs presents significant challenges, it also drives innovation in application design and optimization strategies. By implementing techniques such as context chunking, efficient prompt templates, and graceful fallbacks, developers can mitigate these limitations and enhance the performance and cost-effectiveness of their applications. As the technology evolves, advancements in contextual awareness, token efficiency, and memory management will further empower developers to build more robust and scalable LLM-powered systems. It is crucial to stay informed about these developments and continuously adapt to leverage the full potential of LLMs while addressing their inherent constraints.


## References

- [LangChain Text Splitter](https://langchain.readthedocs.io/en/latest/modules/text_splitter.html).



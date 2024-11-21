# Wrestling with Structured Output

## The Structured Output Challenges

Large language models (LLMs) excel at generating human-like text, but they often struggle to produce output in a structured format consistently. This poses a significant challenge when we need LLMs to generate data that can be easily processed by other systems, such as databases, APIs, or other software applications.  

Sometimes, even with a well-crafted prompt, an LLM might produce an unstructured response when a structured one is expected. This can be particularly challenging when integrating LLMs into systems that require specific data formats.

    ```python
    import openai

    # Define the prompt expecting a structured JSON response
    prompt = """
    Give me the name and capital of 5 random countries in a JSON object with the following keys:
    - name: a string representing the name the country
    - capital: a string representing the capital of the country
    """

    # Call the OpenAI API
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=100
    )

    # Print the response
    print(response.choices[0].text.strip())
    ```

In this example, despite the prompt clearly asking for a JSON object, the LLM generates a natural language sentence instead. This highlights the inconsistency and unpredictability of LLMs when it comes to producing structured output.

## Problem Statement

Obtaining structured output from LLMs presents several significant challenges:

* **Inconsistency**: LLMs often produce unpredictable results, sometimes generating well-structured output and other times deviating from the expected format.
* **Lack of Type Safety**: LLMs do not inherently understand data types, which can lead to errors when their output is integrated with systems requiring specific data formats.
* **Prompt Engineering Complexity**: Crafting prompts that effectively guide LLMs to produce the correct structured output is complex and requires extensive experimentation.

## Solutions

Several strategies and tools can be employed to address the challenges of structured output from LLMs.

### Strategies

* **Schema Guidance**: Providing the LLM with a clear schema or blueprint of the desired output structure helps to constrain its generation and improve consistency. This can be achieved by using tools like Pydantic to define the expected data structure and then using that definition to guide the LLM's output. 
* **Output Parsing**: When LLMs don't natively support structured output, parsing their text output using techniques like regular expressions or dedicated parsing libraries can extract the desired information. For example, you can use regular expressions to extract specific patterns from the LLM's output, or you can use libraries like Pydantic to parse the output into structured data objects.
* **Type Enforcement**: Using tools that enforce data types, such as Pydantic in Python, can ensure that the LLM output adheres to the required data formats. This can help to prevent errors when integrating the LLM's output with other systems.

### Tools

* **One-Shot Prompts**: In one-shot prompting, you provide a single example of the desired output format within the prompt.  While simple, this approach may not be sufficient for complex structures.
    ```python
    prompt = """A "whatpu" is a small, furry animal native to Tanzania. An example of a sentence that uses the word whatpu is:   We were traveling in Africa and we saw these very cute whatpus.   To do a "farduddle" means to jump up and down really fast. An example of a sentence that uses the word farduddle is:"""
    # The LLM should now be able to generate a sentence using "farduddle" correctly.
    ```
* **Gemini-Specific Structured Outputs**: Google's Gemini API offers features specifically designed for generating JSON output. You can provide a schema either within the prompt or through model configuration.  Gemini also supports using enums to restrict the model's output to specific options.
    ```python
    import enum
    from typing_extensions import TypedDict
    import google.generativeai as genai

    class Grade(enum.Enum):
        A_PLUS = "a+"
        A = "a"
        B = "b"
        C = "c"
        D = "d"
        F = "f"

    class Recipe(TypedDict):
        recipe_name: str
        grade: Grade

    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    result = model.generate_content(
        "List about 10 cookie recipes, grade them based on popularity",
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            response_schema=list[Recipe]
        ),
    )
    print(result) 
    # Example Output: [{"grade": "a+", "recipe_name": "Chocolate Chip Cookies"}, ...]
    ```
* **LangChain**:  LangChain is a framework designed to simplify the development of LLM applications. It offers several tools for parsing structured output, including:
    * **`with_structured_output`**: This method is used with LLMs that support structured output APIs, allowing you to enforce a schema directly within the prompt.
    ```python
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field

    class WeatherForecast(BaseModel):
        city: str = Field(description="City for the forecast")
        date: str = Field(description="Date of the forecast")
        condition: str = Field(description="Weather condition")
        temperature: int = Field(description="Temperature in Celsius")

    # Define your LLM
    llm = OpenAI(temperature=0)

    # Define the prompt template
    template = """Provide the weather forecast for the following city: {city}
    Format your response like this:
    ```json
    {{
    "city": "...",
    "date": "...",
    "condition": "...",
    "temperature": ...
    }}
    ```"""
    prompt = PromptTemplate(template=template, input_variables=["city"])

    # Define the output parser
    parser = PydanticOutputParser(pydantic_object=WeatherForecast)

    # Create a chain
    chain = prompt | llm.bind(parser=parser)

    # Run the chain
    result = chain.invoke({"city": "London"})
    print(result)
    ```
    * **`PydanticOutputParser`**: This class leverages Pydantic models to enforce type safety and validate LLM output against a predefined schema.
    ```python
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.output_parsers import PydanticOutputParser
    from pydantic import BaseModel, Field

    class Movie(BaseModel):
        title: str
        director: str
        runtime_minutes: int

    class FilmFestival(BaseModel):
        name: str
        movies: list[Movie]

    # Define your LLM
    llm = OpenAI(temperature=0)

    # Define the prompt template
    template = """Tell me about the movies playing at {festival_name} film festival.
    Provide the title, director, and runtime in minutes for each movie."""
    prompt = PromptTemplate(template=template, input_variables=["festival_name"])

    # Define the output parser
    parser = PydanticOutputParser(pydantic_object=FilmFestival)

    # Create a chain
    chain = prompt | llm.bind(parser=parser)

    # Run the chain
    result = chain.invoke({"festival_name": "Cannes"})
    print(result) 
    ```
    * **`StructuredOutputParser`**: This class offers flexibility in defining custom schemas using `ResponseSchema` objects, making it suitable for extracting information that doesn't fit into pre-built structures. 
    ```python
    from langchain.llms import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.output_parsers import StructuredOutputParser, ResponseSchema

    # Define your LLM
    llm = OpenAI(temperature=0)

    # Define the output parser
    parser = StructuredOutputParser.from_response_schemas([
        ResponseSchema(name="recipe", description="The name of the recipe."),
        ResponseSchema(name="ingredients", description="A list of ingredients."),
    ])

    # Define the prompt template
    format_instructions = parser.get_format_instructions()
    prompt = PromptTemplate(
        template="Provide a recipe and its ingredients. {format_instructions}",
        input_variables=["format_instructions"],
        partial_variables={"format_instructions": format_instructions}
    )

    # Create a chain
    chain = prompt | llm | parser

    # Run the chain
    result = chain.invoke({})
    print(result) 
    ```
* **Outlines**: Outlines is a library specifically focused on structured text generation from LLMs.  It provides several powerful features:
    * **Multiple Choice Generation**: Restrict the LLM output to a predefined set of options.
    ```python
    import outlines

    model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")

    prompt = """You are a sentiment-labelling assistant.

    Is the following review positive or negative?

    Review: This restaurant is just awesome!

    """

    generator = outlines.generate.choice(model, ["Positive", "Negative"])

    answer = generator(prompt)
    ```
    * **Type Constraints**:  Force the output to be integers, floats, or other specific types.
    ```python
    import outlines

    model = outlines.models.transformers("WizardLM/WizardMath-7B-V1.1")

    prompt = "<s>result of 9 + 9 = 18</s><s>result of 1 + 2 = "

    answer = outlines.generate.format(model, int)(prompt)

    print(answer)

    # Output: 3
    ```
    * **Regex-Structured Generation**: Efficiently extract information using regular expressions.
    ```python
    import outlines

    model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")

    prompt = "What is the IP address of the Google DNS servers? "

    generator = outlines.generate.regex(
        model,
        r"((25|2\d|?\d\d?)\.){3}(25|2\d|?\d\d?)",
    )

    structured = generator(prompt, max_tokens=30)

    print(structured)

    # Example Output: 8.8.8.8
    ```
    * **JSON Generation with Pydantic or JSON Schema**: Guarantee that the output conforms to a given Pydantic model or JSON schema.
    ```python
    from enum import Enum
    from pydantic import BaseModel, constr
    import outlines

    class Weapon(str, Enum):
        sword = "sword"
        axe = "axe"
        mace = "mace"
        spear = "spear"
        bow = "bow"
        crossbow = "crossbow"

    class Armor(str, Enum):
        leather = "leather"
        chainmail = "chainmail"
        plate = "plate"

    class Character(BaseModel):
        name: constr(max_length=10)
        age: int
        armor: Armor
        weapon: Weapon
        strength: int

    model = outlines.models.transformers("microsoft/Phi-3-mini-4k-instruct")

    generator = outlines.generate.json(model, Character)

    character = generator("Give me a character description")

    print(repr(character))

    # Example Output: Character(name='Aaliyah', age=35, armor=<Armor.plate: 'plate'>, weapon=<Weapon.bow: 'bow'>, strength=10)
    ```
    * **Context-Free Grammar (CFG) Guidance**: Use formal grammars to define the valid structure of the output.
    ```python
    import outlines

    arithmetic_grammar = """
    ?start: expression
    ?expression: term (("+" | "-") term)*
    ?term: factor (("*" | "/") factor)*
    ?factor: NUMBER
    | "-" factor
    | "(" expression ")"
    %import common.NUMBER
    """

    model = outlines.models.transformers("WizardLM/WizardMath-7B-V1.1")

    generator = outlines.generate.cfg(model, arithmetic_grammar)

    sequence = generator("Alice had 4 apples and Bob ate 2. Write an expression for Alice's apples:")

    print(sequence)

    # Output: 4-2 
    ```

### Comparing Solutions

* **Simplicity vs. Control**: One-shot prompts are simple but offer limited control.  Dedicated tools like Gemini's structured output features, LangChain, and Outlines provide greater control but might have a steeper learning curve.
* **Native LLM Support**:  `with_structured_output` in LangChain relies on the LLM having built-in support for structured output APIs. Other methods, like parsing or using Outlines, are more broadly applicable.
* **Flexibility**:  Outlines and LangChain's  `StructuredOutputParser`  offer the most flexibility for defining custom output structures.

## Best Practices

* **Clear Schema Definition**: Define the desired output structure clearly, using schemas, types, or grammars as appropriate. This ensures the LLM knows exactly what format is expected.
* **Descriptive Naming**: Use meaningful names for fields and elements in your schema. This makes the output more understandable and easier to work with.
* **Detailed Prompting**: Guide the LLM with well-crafted prompts that include examples and clear instructions.  A well-structured prompt improves the chances of getting the desired output.
* **Error Handling**: Implement mechanisms to handle cases where the LLM deviates from the expected structure. LLMs are not perfect, so having error handling ensures your application remains robust.
* **Testing and Iteration**: Thoroughly test your structured output generation process and refine your prompts and schemas based on the results. Continuous testing and refinement are key to achieving reliable structured output. 

## Conclusion

Extracting structured output from LLMs is crucial for integrating them into real-world applications. By understanding the challenges and employing appropriate strategies and tools, developers can improve the reliability and usability of LLM-powered systems, unlocking their potential to automate complex tasks and generate valuable insights. 

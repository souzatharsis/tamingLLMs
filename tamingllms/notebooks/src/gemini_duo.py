from markitdown import MarkItDown
from typing import List, Dict, Tuple

import os
import google.generativeai as genai
from google.generativeai import caching
import datetime
from langchain import hub

from dotenv import load_dotenv
import os


PROMPT_TEMPLATE = "souzatharsis/duo_msg"
PROMPT_COMMIT = "2f61756a" #"75e56f77" 
PROMPT_QUIZ_TEMPLATE = "souzatharsis/duo_quiz"
PROMPT_QUIZ_COMMIT = "9e9a7500"

GEMINI_MODEL_NAME = "gemini-1.5-pro-002"
GEMINI_API_KEY_LABEL = "GEMINI_API_KEY"
GEMINI_GROUNDED_MODEL_NAME = "gemini-1.5-flash-002"

# Load environment variables from .env file
load_dotenv()

class LLMBackend:
    """
    A backend class for managing LLM interactions.
    """
    CACHE_TTL = 60 # cache time-to-live in minutes
    def __init__(
        self,
        model_name: str = GEMINI_MODEL_NAME,
        api_key_label: str = GEMINI_API_KEY_LABEL,
        conversation_config: Dict = {},
        input: str = "",
        cache_ttl: int = CACHE_TTL,
    ):
        """
        Initialize the LLMBackend.

        Args:
                temperature (float): The temperature for text generation.
                model_name (str): The name of the model to use.
        """
        self.model_name = model_name
        
        #secret_value = UserSecretsClient().get_secret(api_key_label)#TODO
        genai.configure(api_key=os.environ[api_key_label])
        

        self.cache = caching.CachedContent.create(
            model=model_name,
            display_name='due_knowledge_base', # used to identify the cache
            system_instruction=(
                self.compose_prompt(input, conversation_config)
            ),
            ttl=datetime.timedelta(minutes=cache_ttl),
        )

        
        
        self.model = genai.GenerativeModel.from_cached_content(cached_content=self.cache)
    
    def compose_prompt(self, input:str, conversation_config: Dict) -> str:
        """
        Compose a prompt for the Gemini Duo model using a LangChain prompt template.
        """
        prompt_template = hub.pull(f"{PROMPT_TEMPLATE}:{PROMPT_COMMIT}")
        prompt = prompt_template.invoke({"memory": input,
          "input_texts": ""})
        
        return prompt.messages[0].content


class Quiz:
    """
    A backend class for managing quiz generation.
    """
    CACHE_TTL = 60 # cache time-to-live in minutes
    def __init__(
        self,
        model_name: str = GEMINI_MODEL_NAME,
        api_key_label: str = GEMINI_API_KEY_LABEL,
        input: str = "",
        cache_ttl: int = CACHE_TTL,
        add_citations: bool = False,
        num_questions: int = 10,
    ):
        """
        Initialize the Quiz.

        Args:

        """
        self.model_name = model_name
        self.input = input
        #secret_value = UserSecretsClient().get_secret(api_key_label)#TODO
        genai.configure(api_key=os.environ[api_key_label])

        self.cache = caching.CachedContent.create(
            model=model_name,
            display_name='due_quiz', # used to identify the cache
            system_instruction=(
                self.compose_prompt(num_questions, add_citations)),
                contents=input,
            ttl=datetime.timedelta(minutes=cache_ttl),
        )

        
        
        self.model = genai.GenerativeModel.from_cached_content(cached_content=self.cache)

    def compose_prompt(self, num_questions: int = 10, add_citations: bool = True) -> str:
        if add_citations:
            citations = "In the Quiz answers, at the end, make sure to add Input ID indicating referenced content pertaining to the answer to the question."
        else:
            citations = ""
        prompt_template = hub.pull(f"{PROMPT_QUIZ_TEMPLATE}:{PROMPT_QUIZ_COMMIT}")
        prompt = prompt_template.invoke({"num_questions": num_questions,
          "citations": citations,
          "input_texts": ""})
        return prompt.messages[0].content

    
    def generate(self, msg:str="") -> str:
        msg = f"Generate: {msg}"
        response = self.model.generate_content([msg])
        return response


class ContentGenerator:
    """
    A class to handle content generation using the Gemini model.
    """
    def __init__(self, model):
        """
        Initialize the ContentGenerator with a model and input content.
        
        Args:
            model: The Gemini model instance to use for generation
            input_content: The input content to generate from
        """
        self.cached_model = model
        self.non_cached_model = None

    # cached generation
    def generate(self, input_content, user_instructions=""):
        """
        Generate content using the model.
        
        Returns:
            The model's response
        """
        prompt=f"""
        USER_INSTRUCTIONS: Make sure to follow these instructions: {user_instructions}
        INPUT:{input_content}
        """

        response = self.cached_model.generate_content([(
            prompt)])
        return response


class Client:

    def __init__(self, knowledge_base: List[str] = []):
        """
        Initialize the Client class with conversation configuration.
        """
        self.knowledge_base = knowledge_base # user-provided foundation knowledge represented as a list of urls
        self.reference_id = 0 # unique ID for each input
        self.input = "" # short-term memory, i.e. current input to be studied
        self.urls = [] # input list of URLs to extract content from
        self.response = "" # latest response from LLM
        self.urls_memory = [] # cumulative list of URLs to extract content from
        self.input_memory = "" # long-term memory, i.e. cumulative input + knowledge base
        self.response_memory = "" # long-term response memory, i.e. cumulative responses
        self.extractor = MarkItDown() # extractor for content from URLs

        self.quiz_instance = None

        self.add_knowledge_base(self.knowledge_base) 

        self.llm = LLMBackend(input=self.input_memory
                              ) # llm with cached content
        
        self.content_generator = ContentGenerator(model=self.llm.model) # content 

    def add_knowledge_base(self, urls: List[str]) -> None:
        """
        Add URLs to the knowledge base.
        """
        self.add(urls)

    def add(self, urls: List[str]) -> None:
        """
        Extract content from URLs and add it to the conversation input.

        Args:
            urls (List[str]): List of URLs to extract content from.
            refocus (bool): Whether to clear current conversation state before adding.
                          Defaults to True.
        """

        self.urls = urls

        # Add new content to input following CIC format to enable citations
        for url in urls:
            self.urls_memory.append(url)
            content = self.extractor.convert(url).text_content
            formatted_content = f"ID: {self.reference_id} | {content} | END ID: {self.reference_id}"
            self.input += formatted_content + "\n" 
            self.reference_id += 1
        
        # Update memory
        self.input_memory = self.input_memory + self.input
    
    def msg(self, msg: str = "", add_citations: bool = False) -> str:
        """
        Generate Q&A content based on the current conversation input.

        Args:
            msg (str): Additional instructions for content generation.
            longform (bool): Whether to generate detailed, long-form content.
                           Defaults to False.
            add_citations (bool): Whether to include input citations in responses.
                                Defaults to False.
            transcript_only (bool): Whether to only return the transcript.
                                Defaults to False.

        Returns:
            str: Generated Q&A conversation content.
        """
        if add_citations:
            msg = msg + "\n\n For key statements, add Input ID to the response."

        self.response = self.content_generator.generate(
            input_content=self.input,
            user_instructions=msg
        )
        print(self.response.usage_metadata)

        self.response_memory = self.response_memory + self.response.text

        return self.response.text

    def quiz(self, add_citations: bool = True, num_questions: int = 10) -> str:
        """
        Generate a quiz based on full input memory.
        """
        self.quiz_instance = Quiz(
                         input=self.input_memory,
                         add_citations=add_citations,
                         num_questions=num_questions)
        #response = self.quiz_instance.generate(msg)
        #print(response.usage_metadata)
        #return response.text
        return self.quiz_instance
    




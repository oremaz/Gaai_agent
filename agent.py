from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.readers.file import PDFReader, DocxReader, CSVReader, ImageReader
import os
from typing import List, Dict, Any
from llama_index.tools.arxiv import ArxivToolSpec
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
import re
from llama_index.core.agent.workflow import ReActAgent
import wandb
from llama_index.callbacks.wandb import WandbCallbackHandler
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.llama_debug import LlamaDebugHandler
from llama_index.core import Settings

from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM
import requests
import logging
from llama_index.core.workflow import Context
from llama_index.core.agent.workflow import AgentStream
from llama_index.readers_web import TrafilaturaWebReader
from llama_index_readers_youtube_transcript import YoutubeTranscriptReader



wandb_callback = WandbCallbackHandler(run_args={"project": "gaia-llamaindex-agents"})
llama_debug = LlamaDebugHandler(print_trace_on_end=True)

# Comprehensive callback manager
callback_manager = CallbackManager([
    wandb_callback,     # For W&B tracking
    llama_debug        # For general debugging
])

logging.basicConfig(level=logging.INFO)
logging.getLogger("llama_index.core.agent").setLevel(logging.DEBUG)
logging.getLogger("llama_index.llms").setLevel(logging.DEBUG)

model_id = "Qwen/Qwen2.5-7B-Instruct" 
proj_llm = HuggingFaceLLM(
    model_name=model_id,
    tokenizer_name=model_id,
    device_map="auto",           # will use GPU if available
    model_kwargs={"torch_dtype": "auto"},
    generate_kwargs={"temperature": 0.1, "top_p": 0.3}  # More focused
)

embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")

wandb.init(project="gaia-llamaindex-agents")  # Choisis ton nom de projet
wandb_callback = WandbCallbackHandler(run_args={"project": "gaia-llamaindex-agents"})
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([wandb_callback, llama_debug])

Settings.llm = proj_llm
Settings.embed_model = embed_model
Settings.callback_manager = callback_manager

import os
from typing import List
from urllib.parse import urlparse

from llama_index.core.tools import FunctionTool
from llama_index.core import Document

# --- Import all required official LlamaIndex Readers ---
from llama_index.readers.file import (
    PDFReader,
    DocxReader,
    CSVReader,
    PandasExcelReader,
    ImageReader,
)
from llama_index.readers.json import JSONReader
from llama_index.readers.web import TrafilaturaWebReader
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
from llama_index.readers.audiotranscribe.openai import OpenAIAudioTranscriptReader

def read_and_parse_content(input_path: str) -> List[Document]:
    """
    Reads and parses content from a file path or URL into Document objects.
    It automatically detects the input type and uses the appropriate LlamaIndex reader.

    Args:
        input_path: A local file path or a web URL.

    Returns:
        A list of LlamaIndex Document objects with the extracted text.
    """
    # --- Completed readers map for various local file types ---
    readers_map = {
        # Documents
        '.pdf': PDFReader(),
        '.docx': DocxReader(),
        '.doc': DocxReader(),
        # Data files
        '.csv': CSVReader(),
        '.json': JSONReader(),
        '.xlsx': PandasExcelReader(),
        # Media files
        '.jpg': ImageReader(),
        '.jpeg': ImageReader(),
        '.png': ImageReader(),
        '.mp3': OpenAIAudioTranscriptReader(),
    }

    # --- URL Handling ---
    if input_path.startswith("http"):
        if "https://www.youtube.com/watch?v=2N-rwsa5lEw2" in urlparse(input_path).netloc or "https://www.youtube.com/watch?v=2N-rwsa5lEw3" in urlparse(input_path).netloc:
            loader = YoutubeTranscriptReader()
            documents = loader.load_data(youtubelinks=[input_path])
        else:
            loader = TrafilaturaWebReader()
            documents = loader.load_data(urls=[input_path])
    
    # --- Local File Handling ---
    else:
        if not os.path.exists(input_path):
            return [Document(text=f"Error: File not found at {input_path}")]
        
        file_extension = os.path.splitext(input_path)[1].lower()

        if file_extension in readers_map:
            loader = readers_map[file_extension]
            documents = loader.load_data(file=input_path)
        else:
            # Fallback for text-based files without a specific reader (e.g., .py, .txt, .md)
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                documents = [Document(text=content, metadata={"source": input_path})]
            except Exception as e:
                return [Document(text=f"Error reading file as plain text: {e}")]
    
    # Add the source path to metadata for traceability
    for doc in documents:
        doc.metadata["source"] = input_path
        
    return documents

# --- Create the final LlamaIndex Tool from the completed function ---
read_and_parse_tool = FunctionTool.from_defaults(
    fn=read_and_parse_content,
    name="read_and_parse_tool",
    description=(
        "Use this tool to read and extract content from any given file or URL. "
        "It handles PDF, DOCX, CSV, JSON, XLSX, and image files, as well as web pages, "
        "YouTube videos (transcripts), and MP3 audio (transcripts). It also reads plain text "
        "from files like .py or .txt. The input MUST be a single valid file path or a URL."
    )
)

from typing import List
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.tools import QueryEngineTool
from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine

def create_rag_tool(documents: List[Document]) -> QueryEngineTool:
    """
    Creates a RAG query engine tool from a list of documents using advanced components.
    Inspired by 'create_advanced_index' and 'create_context_aware_query_engine' methods.

    Args:
        documents: A list of LlamaIndex Document objects from the read_and_parse_tool.

    Returns:
        A QueryEngineTool configured for the agent to use in the current task.
    """
    if not documents:
        return None

    # --- 1. Node Parsing (from your 'create_advanced_index' logic) ---
    # Using the exact parsers and logic you defined.
    hierarchical_parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
    sentence_window_parser = SentenceWindowNodeParser.from_defaults(
        window_size=3,
        window_metadata_key="window",
        original_text_metadata_key="original_text",
    )
    
    # Choose parser based on document count
    if len(documents) > 5: # Heuristic for using hierarchical parser
        nodes = hierarchical_parser.get_nodes_from_documents(documents)
    else:
        nodes = sentence_window_parser.get_nodes_from_documents(documents)

    # --- 2. Index Creation ---
    # Assumes Settings.embed_model is configured globally as in your snippet
    index = VectorStoreIndex(nodes)

    # --- 3. Query Engine Creation (from your 'create_context_aware_query_engine' logic) ---
    # Using the exact reranker you specified
    reranker = SentenceTransformerRerank(
        model="cross-encoder/ms-marco-MiniLM-L-2-v2", 
        top_n=5
    )
    
    query_engine = index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[reranker],
        # Assumes Settings.llm is configured globally
    )
    
    # --- 4. Wrap the Query Engine in a Tool ---
    rag_engine_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="rag_engine_tool",
        description=(
            "Use this tool to ask questions and query the content of documents that have already "
            "been loaded. This is your primary way to find answers from the provided context. "
            "The input is a natural language question about the documents' content."
        )
    )
    
    return rag_engine_tool


import re
from llama_index.core.tools import FunctionTool
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec

# 1. Create the base DuckDuckGo search tool from the official spec.
# This tool returns text summaries of search results, not just URLs.
base_duckduckgo_tool = DuckDuckGoSearchToolSpec().to_tool_list()[0]

# 2. Define a wrapper function to post-process the output.
def search_and_extract_top_url(query: str) -> str:
    """
    Takes a search query, uses the base DuckDuckGo search tool to get results,
    and then parses the output to extract and return only the first URL.

    Args:
        query: The natural language search query.

    Returns:
        A string containing the first URL found, or an error message if none is found.
    """
    # Call the base tool to get the search results as text
    search_results = base_duckduckgo_tool(query)
    
    # Use a regular expression to find the first URL in the text output
    # The \S+ pattern matches any sequence of non-whitespace characters
    url_match = re.search(r"https?://\S+", str(search_results))
    
    if url_match:
        return url_match.group(0)
    else:
        return "No URL could be extracted from the search results."

# 3. Create the final, customized FunctionTool for the agent.
# This is the tool you will actually give to your agent.
extract_url_tool = FunctionTool.from_defaults(
    fn=search_and_extract_top_url,
    name="extract_url_tool",
    description=(
        "Use this tool ONLY when you need to find a relevant URL to answer a question but no "
        "specific file, document, or URL has been provided. It takes a search query as input "
        "and returns a single, relevant URL."
    )
)

def execute_python_code(code: str) -> str:
    try:
        safe_globals = {
            "__builtins__": {
                "len": len, "str": str, "int": int, "float": float,
                "list": list, "dict": dict, "sum": sum, "max": max, "min": min,
                "round": round, "abs": abs, "sorted": sorted, "enumerate": enumerate,
                "range": range, "zip": zip, "map": map, "filter": filter,
                "any": any, "all": all, "type": type, "isinstance": isinstance,
                "print": print, "open": open, "bool": bool, "set": set, "tuple": tuple
            },
            # Core Python modules
            "math": __import__("math"),
            "datetime": __import__("datetime"),
            "re": __import__("re"),
            "os": __import__("os"),
            "sys": __import__("sys"),
            "json": __import__("json"),
            "csv": __import__("csv"),
            "random": __import__("random"),
            "itertools": __import__("itertools"),
            "collections": __import__("collections"),
            "functools": __import__("functools"),
            
            # Data Science and Numerical Computing
            "numpy": __import__("numpy"),
            "np": __import__("numpy"),
            "pandas": __import__("pandas"),
            "pd": __import__("pandas"),
            "scipy": __import__("scipy"),
            
            # Visualization
            "matplotlib": __import__("matplotlib"),
            "plt": __import__("matplotlib.pyplot"),
            "seaborn": __import__("seaborn"),
            "sns": __import__("seaborn"),
            "plotly": __import__("plotly"),
            
            # Machine Learning
            "sklearn": __import__("sklearn"),
            "xgboost": __import__("xgboost"),
            "lightgbm": __import__("lightgbm"),
            
            # Statistics
            "statistics": __import__("statistics"),
            "statsmodels": __import__("statsmodels"),
            
            # Image Processing
            "PIL": __import__("PIL"),
            "cv2": __import__("cv2"),
            "skimage": __import__("skimage"),
            
            # Network and Web
            "requests": __import__("requests"),
            "urllib": __import__("urllib"),
            
            # Text Processing
            "nltk": __import__("nltk"),
            "spacy": __import__("spacy"),
            
            # Time Series
            "pytz": __import__("pytz"),
            
            # Utilities
            "tqdm": __import__("tqdm"),
            "pickle": __import__("pickle"),
            "gzip": __import__("gzip"),
            "base64": __import__("base64"),
            "hashlib": __import__("hashlib"),
            "uuid": __import__("uuid"),
            
            # Scientific Computing
            "sympy": __import__("sympy"),
            "networkx": __import__("networkx"),
            
            # Database
            "sqlite3": __import__("sqlite3"),
            
            # Parallel Processing
            "multiprocessing": __import__("multiprocessing"),
            "threading": __import__("threading"),
            "concurrent": __import__("concurrent"),
        }
            
        exec_locals = {}
        exec(code, safe_globals, exec_locals)
    
        if 'result' in exec_locals:
            return str(exec_locals['result'])
        else:
            return "Code executed successfully"
    
    except Exception as e:
        return f"Code execution failed: {str(e)}"

code_execution_tool = FunctionTool.from_defaults(
fn=execute_python_code,
name="Python Code Execution",
description="Execute Python code safely for calculations and data processing"
)

import re
from llama_index.core.tools import FunctionTool
from llama_index.llms.huggingface import HuggingFaceLLM

# --- 1. Initialize a dedicated LLM for Code Generation ---
# It's good practice to use a model specifically fine-tuned for coding.
# This model is loaded only once for efficiency.
try:
    code_llm = HuggingFaceLLM(
        model_name="Qwen/Qwen2.5-Coder-7B",
        tokenizer_name="Qwen/Qwen2.5-Coder-7B",
        device_map="auto",
        model_kwargs={"torch_dtype": "auto"},
        # Set generation parameters for precise, non-creative code output
        generate_kwargs={"temperature": 0.0, "do_sample": False}
    )
except Exception as e:
    print(f"Error initializing code generation model: {e}")
    print("Code generation tool will not be available.")
    code_llm = None


def generate_python_code(query: str) -> str:
    """
    Generates executable Python code based on a natural language query.

    Args:
        query: A detailed description of the desired functionality for the Python code.

    Returns:
        A string containing only the generated Python code, ready for execution.
    """
    if not code_llm:
        return "Error: Code generation model is not available."

    # --- 2. Create a precise prompt for the code model ---
    # This prompt explicitly asks for only code, no explanations.
    prompt = f"""
Your task is to generate ONLY the Python code for the following request.
Do not include any explanations, introductory text, or markdown formatting like '```python'.
The output must be a single, clean block of Python code.

Request: "{query}"

Python Code:
"""

    # --- 3. Generate the response and post-process it ---
    response = code_llm.complete(prompt)
    raw_code = str(response)

    # --- 4. Clean the output to ensure it's pure code ---
    # Models often wrap code in markdown fences, this removes them.
    code_match = re.search(r"```(?:python)?\n(.*)```", raw_code, re.DOTALL)
    if code_match:
        # Extract the code from within the markdown block
        return code_match.group(1).strip()
    else:
        # If no markdown, assume the model followed instructions and return the text directly
        return raw_code.strip()


# --- 5. Create the LlamaIndex Tool from the function ---
generate_code_tool = FunctionTool.from_defaults(
    fn=generate_python_code,
    name="generate_python_code_tool",
    description=(
        "Use this tool to generate executable Python code based on a natural language description of a task. "
        "The input should be a clear and specific request for what the code should do (e.g., 'a function to "
        "calculate the nth Fibonacci number'). The tool returns a string containing only the Python code."
    )
)


class EnhancedGAIAAgent:
    def __init__(self):
        print("Initializing Enhanced GAIA Agent...")
        
        # Vérification du token HuggingFace
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable is required")

        # Agent coordinateur principal qui utilise les agents spécialisés comme tools
        self.coordinator = ReActAgent(
            name="GAIACoordinator",
            description="Main GAIA coordinator that uses specialized capabilities as intelligent tools",
            system_prompt="""
            You are the main GAIA coordinator using ReAct reasoning methodology.
                        
            You have access to THREE specialist tools:
            
            **1. analysis_tool** - Advanced multimodal document analysis specialist
            - Use for: PDF, Word, CSV, image file analysis
            - When to use: Questions with file attachments, document analysis, data extraction
            
            **2. research_tool** - Intelligent research specialist with automatic routing
            - Use for: External knowledge, current events, scientific papers
            - When to use: Questions requiring external knowledge, factual verification, current information
            
            **3. code_tool** - Advanced computational specialist using ReAct reasoning
            - Use for: Mathematical calculations, data processing, logical operations
            - Capabilities: Generates and executes Python, handles complex computations, step-by-step problem solving
            - When to use: Precise calculations, data manipulation, mathematical problem solving

            **4. code_execution_tool** - Use only to execute .py file
                       
            CRITICAL: Your final answer must be EXACT and CONCISE as required by GAIA format : NO explanations, NO additional text, ONLY the precise answer
            """,
            llm=proj_llm,
            tools=[analysis_tool, research_tool, code_tool, code_execution_tool], 
            max_steps=10, 
            verbose = True, 
            callback_manager=callback_manager,

        )

    async def format_gaia_answer(self, raw_response: str, original_question: str) -> str:
        """
        Post-process the agent response to extract the exact GAIA format answer
        """
        format_prompt = f"""Extract the exact answer from the response below. Follow GAIA formatting rules strictly.
    
    Examples:
    
    Question: "How many research papers were published by the university between 2010 and 2020?"
    Response: "Based on my analysis of the data, I found that the university published 156 research papers between 2010 and 2020."
    Answer: 156
    
    Question: "What is the last name of the software engineer mentioned in the report?"
    Response: "After reviewing the document, the software engineer mentioned is Dr. Martinez who developed the system."
    Answer: Martinez
    
    Question: "List the programming languages from this job description, alphabetized:"
    Response: "The job description mentions several programming languages including Python, Java, C++, and JavaScript. When alphabetized, these are: C++, Java, JavaScript, Python"
    Answer: C++, Java, JavaScript, Python
    
    Question: "Give only the first name of the developer who created the framework."
    Response: "The framework was created by Sarah Johnson, a senior developer at the company."
    Answer: Sarah
    
    Question: "Give the ISO country code as your answer."
    Response: "The country in question is France, which has the ISO code FRA."
    Answer: FRA
    
    Question: "Provide your response in standard notation."
    Response: "The calculated value is 314 million, which in standard notation is 3.14e+8"
    Answer: 3.14e+8
    
    Now extract the exact answer:
    
    Question: {original_question}
    Response: {raw_response}
    Answer:"""
    
        try:
            formatting_response = proj_llm.complete(format_prompt)
            answer = str(formatting_response).strip()
            
            # Extract just the answer after "Answer:"
            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()
            
            return answer
            
        except Exception as e:
            print(f"Error in formatting: {e}")
            return self._extract_fallback_answer(raw_response)
    
    def download_gaia_file(self, task_id: str, api_url: str = "https://agents-course-unit4-scoring.hf.space") -> str:
        """Download file associated with task_id"""
        try:
            response = requests.get(f"{api_url}/files/{task_id}", timeout=30)
            response.raise_for_status()
            
            # Save file locally
            filename = f"task_{task_id}_file"
            with open(filename, 'wb') as f:
                f.write(response.content)
            return filename
        except Exception as e:
            print(f"Failed to download file for task {task_id}: {e}")
            return None

    async def solve_gaia_question(self, question_data: Dict[str, Any]) -> str:
            question = question_data.get("Question", "")
            task_id = question_data.get("task_id", "")
    
            # Try to download file
            try:
                file_path = self.download_gaia_file(task_id)
            except Exception as e:
                print(f"Failed to download file for task {task_id}: {e}")
                file_path = None
    
            context_prompt = f"""
            GAIA Task ID: {task_id}
            Question: {question}
            {'File downloaded: ' + file_path if file_path else 'No additional files referenced'}
            
            Additionnal instructions to system prompt :
            1. If a file is available, use the analysis_tool (except for .py files).
            2. If a link is in the question, use the research_tool.
            """
            
            try:
                ctx = Context(self.coordinator)
                
                # Use streaming to see step-by-step reasoning
                print("=== AGENT REASONING STEPS ===")
                handler = self.coordinator.run(ctx=ctx, user_msg=context_prompt)
                
                full_response = ""
                async for event in handler.stream_events():
                    if isinstance(event, AgentStream):
                        print(event.delta, end="", flush=True)
                        full_response += event.delta
                
                # Get the final response
                raw_response = await handler
                print("\n=== END REASONING ===")
                
                # Post-process to extract exact GAIA format
                formatted_answer = await self.format_gaia_answer(str(raw_response), question)
                
                print(f"Formatted answer: {formatted_answer}")
                
                return formatted_answer
                
            except Exception as e:
                error_msg = f"Error processing question: {str(e)}"
                print(error_msg)
                return error_msg
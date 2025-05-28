# Standard library imports
import logging
import os
import re
from typing import Dict, Any, List
from urllib.parse import urlparse

# Third-party imports
import requests
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer

# LlamaIndex core imports
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.agent.workflow import FunctionAgent, ReActAgent, AgentStream
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.llama_debug import LlamaDebugHandler
from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context

# LlamaIndex specialized imports
from llama_index.callbacks.wandb import WandbCallbackHandler
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.readers.assemblyai import AssemblyAIAudioTranscriptReader
from llama_index.readers.file import PDFReader, DocxReader, CSVReader, ImageReader, PandasExcelReader
from llama_index.readers.json import JSONReader
from llama_index.readers.web import TrafilaturaWebReader
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
from llama_index.tools.arxiv import ArxivToolSpec
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec

# --- Import all required official LlamaIndex Readers ---
from llama_index.readers.file import (
    PDFReader,
    DocxReader,
    CSVReader,
    PandasExcelReader,
    ImageReader,
)
from typing import List, Union
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.tools import QueryEngineTool
from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.query_pipeline import QueryPipeline


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

def get_max_memory_config(max_memory_per_gpu):
    """Generate max_memory config for available GPUs"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        max_memory = {}
        for i in range(num_gpus):
            max_memory[i] = max_memory_per_gpu
        return max_memory
    return None

model_id = "google/gemma-3-12b-it"
proj_llm = HuggingFaceLLM(
    model_name=model_id,
    tokenizer_name=model_id,
    device_map="auto",
    model_kwargs={
        "torch_dtype": "auto",
        "max_memory": get_max_memory_config("10GB")
    },
    generate_kwargs={"temperature": 0.1, "top_p": 0.3}  # More focused
)

code_llm = HuggingFaceLLM(
    model_name="Qwen/Qwen2.5-Coder-3B",
    tokenizer_name="Qwen/Qwen2.5-Coder-3B",
    device_map="auto",
    model_kwargs={
        "torch_dtype": "auto",
        "max_memory": get_max_memory_config("3GB")
    },
    # Set generation parameters for precise, non-creative code output
    generate_kwargs={"temperature": 0.0, "do_sample": False}
)


embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")

wandb.init(project="gaia-llamaindex-agents")  # Choisis ton nom de projet
wandb_callback = WandbCallbackHandler(run_args={"project": "gaia-llamaindex-agents"})
llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([wandb_callback, llama_debug])

Settings.llm = proj_llm
Settings.embed_model = embed_model
Settings.callback_manager = callback_manager


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
        '.mp3': AssemblyAIAudioTranscriptReader(input_path),
    }

    # --- URL Handling ---
    if input_path.startswith("http"):
        if "youtube" in urlparse(input_path):
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
extract_url_tool = FunctionTool.from_defaults(
    fn=search_and_extract_top_url,
    name="extract_url_tool",
    description="Searches web and returns a relevant URL based on a query"
)

def create_rag_tool_fn(documents: List[Document], query: str = None) -> Union[QueryEngineTool, str]:
    """
    Creates a RAG query engine tool from documents with advanced indexing and querying capabilities.
    
    This function implements a sophisticated RAG pipeline using hierarchical or sentence-window parsing
    depending on document count, vector indexing, and reranking for optimal information retrieval.
    
    Args:
        documents (List[Document]): A list of LlamaIndex Document objects from read_and_parse_tool.
                                   Must not be empty to create a valid RAG engine.
        query (str, optional): If provided, immediately queries the created RAG engine and returns
                              the answer as a string. If None, returns the QueryEngineTool for later use.
                              Defaults to None.
    
    Returns:
        Union[QueryEngineTool, str]: 
            - QueryEngineTool: When query=None, returns a tool configured for agent use with
                              advanced reranking and similarity search capabilities.
            - str: When query is provided, returns the direct answer from the RAG engine.
            - None: When documents list is empty.
    
    Examples:
        Create a RAG tool for later use:
        >>> rag_tool = create_rag_tool_fn(documents)
        
        Get immediate answer from documents:
        >>> answer = create_rag_tool_fn(documents, query="What is the main topic?")
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

    if query : 
        result = rag_engine_tool.query_engine.query(query)
        return str(result)
    
    return rag_engine_tool

def information_retrieval_fn (paths : List[str],  query : str = None) -> Union[QueryEngineTool, str]:
    docs = []
    for path in paths : 
        docs.append(read_and_parse_content(path))
    return create_rag_tool_fn(docs,query)
    
information_retrieval_tool = FunctionTool.from_defaults(
    fn=information_retrieval_fn,
    name="information_retrieval_tool",
    description="Retrieves and queries information from documents, URLs, or files using RAG"
)

# 1. Create the base DuckDuckGo search tool from the official spec.
# This tool returns text summaries of search results, not just URLs.
base_duckduckgo_tool = DuckDuckGoSearchToolSpec().to_tool_list()[1]

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
    search_results = base_duckduckgo_tool(query, max_results = 1)
    print(search_results)
    
    # Use a regular expression to find the first URL in the text output
    # The \S+ pattern matches any sequence of non-whitespace characters
    url_match = re.search(r"https?://\S+", str(search_results))
    
    if url_match:
        return url_match.group(0)[:-2]
    else:
        return "No URL could be extracted from the search results."


# Create external_knowledge agent - ReAct agent with extract_url_tool and information_retrieval tool
external_knowledge_agent = ReActAgent(
    name="external_knowledge_agent", 
    description="Retrieves information from external sources and documents",
    system_prompt="You are an information retrieval specialist. You find and extract relevant information from external sources, URLs, and documents to answer queries.""",
    tools=[extract_url_tool, information_retrieval_tool],
    llm=proj_llm,
    max_steps=6,
    verbose=True,
    callback_manager=callback_manager,
)

# 3. Create the final, customized FunctionTool for the agent.
# This is the tool you will actually give to your agent.
extract_url_tool = FunctionTool.from_defaults(
    fn=search_and_extract_top_url,
    name="extract_url_tool",
    description=(
        "Use this tool when you need to find a relevant URL to answer a question. It takes a search query as input and returns a single, relevant URL."
    )
)

import importlib.util
import sys

def safe_import(module_name):
    """Safely import a module, return None if not available"""
    try:
        return __import__(module_name)
    except ImportError:
        return None

safe_globals = {
    "__builtins__": {
        "len": len, "str": str, "int": int, "float": float,
        "list": list, "dict": dict, "sum": sum, "max": max, "min": min,
        "round": round, "abs": abs, "sorted": sorted, "enumerate": enumerate,
        "range": range, "zip": zip, "map": map, "filter": filter,
        "any": any, "all": all, "type": type, "isinstance": isinstance,
        "print": print, "open": open, "bool": bool, "set": set, "tuple": tuple
    }
}

# Core modules (always available)
core_modules = [
    "math", "datetime", "re", "os", "sys", "json", "csv", "random",
    "itertools", "collections", "functools", "operator", "copy",
    "decimal", "fractions", "uuid", "typing", "statistics", "pathlib",
    "glob", "shutil", "tempfile", "pickle", "gzip", "zipfile", "tarfile",
    "base64", "hashlib", "secrets", "hmac", "textwrap", "string",
    "difflib", "socket", "ipaddress", "logging", "warnings", "traceback",
    "pprint", "threading", "queue", "sqlite3", "urllib", "html", "xml",
    "configparser"
]

for module in core_modules:
    imported = safe_import(module)
    if imported:
        safe_globals[module] = imported

# Data science modules (may not be available)
optional_modules = {
    "numpy": "numpy",
    "np": "numpy", 
    "pandas": "pandas",
    "pd": "pandas",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "plt": "matplotlib.pyplot",
    "seaborn": "seaborn",
    "sns": "seaborn",
    "plotly": "plotly",
    "sklearn": "sklearn",
    "statsmodels": "statsmodels",
    "PIL": "PIL",
    "skimage": "skimage",
    "pytz": "pytz",
    "requests": "requests",
    "bs4": "bs4",
    "sympy": "sympy",
    "tqdm": "tqdm",
    "yaml": "yaml",
    "toml": "toml"
}

for alias, module_name in optional_modules.items():
    imported = safe_import(module_name)
    if imported:
        safe_globals[alias] = imported

# Special cases
if safe_globals.get("bs4"):
    safe_globals["BeautifulSoup"] = safe_globals["bs4"].BeautifulSoup

if safe_globals.get("PIL"):
    image_module = safe_import("PIL.Image")
    if image_module:
        safe_globals["Image"] = image_module

def execute_python_code(code: str) -> str:
    try: 
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
    description="Executes Python code safely for calculations and data processing"
)

code_agent = ReActAgent(
    name="code_agent",
    description="Handles Python code for calculations and data processing",
    system_prompt="You are a Python programming specialist. You work with Python code to perform calculations, data analysis, and mathematical operations.",
    tools=[code_execution_tool],
    llm=code_llm,
    max_steps=6,
    verbose=True,
    callback_manager=callback_manager,
)

def clean_response(response: str) -> str:
    """Clean response by removing common prefixes"""
    response_clean = response.strip()
    prefixes_to_remove = [
        "FINAL ANSWER:", "Answer:", "The answer is:", 
        "Based on my analysis,", "After reviewing,", 
        "The result is:", "Final result:", "According to",
        "In conclusion,", "Therefore,", "Thus,"
    ]
    
    for prefix in prefixes_to_remove:
        if response_clean.startswith(prefix):
            response_clean = response_clean[len(prefix):].strip()
    
    return response_clean

def llm_reformat(response: str, question: str) -> str:
    """Use LLM to reformat the response according to GAIA requirements"""
    
    format_prompt = f"""Extract the exact answer from the response below. Follow GAIA formatting rules strictly.

GAIA Format Rules:
- ONLY the precise answer, no explanations
- No prefixes like "Answer:", "The result is:", etc.
- For numbers: just the number (e.g., "156", "3.14e+8")
- For names: just the name (e.g., "Martinez", "Sarah")
- For lists: comma-separated (e.g., "C++, Java, Python")
- For country codes: just the code (e.g., "FRA", "US")
- For yes/no: just "Yes" or "No"

Examples:
Question: "How many papers were published?"
Response: "The analysis shows 156 papers were published in total."
Answer: 156

Question: "What is the last name of the developer?"
Response: "The developer mentioned is Dr. Sarah Martinez from the AI team."
Answer: Martinez

Question: "List programming languages, alphabetized:"
Response: "The languages mentioned are Python, Java, and C++. Alphabetized: C++, Java, Python"
Answer: C++, Java, Python

Now extract the exact answer:
Question: {question}
Response: {response}
Answer:"""

    try:
        # Use the global LLM instance
        formatting_response = proj_llm.complete(format_prompt)
        answer = str(formatting_response).strip()
        
        # Extract just the answer after "Answer:"
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        return answer
    except Exception as e:
        print(f"LLM reformatting failed: {e}")
        return response

def final_answer_tool(agent_response: str, question: str) -> str:
    """
    Simplified final answer tool using only LLM reformatting.
    
    Args:
        agent_response: The raw response from agent reasoning
        question: The original question for context
        
    Returns:
        Exact answer in GAIA format
    """
    
    # Step 1: Clean the response
    cleaned_response = clean_response(agent_response)
    
    # Step 2: Use LLM reformatting
    formatted_answer = llm_reformat(cleaned_response, question)
    
    print(f"Original response cleaned: {cleaned_response[:100]}...")
    print(f"LLM formatted answer: {formatted_answer}")
    
    return formatted_answer


class EnhancedGAIAAgent:
    def __init__(self):
        print("Initializing Enhanced GAIA Agent...")
        
        # Vérification du token HuggingFace
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            print("Warning: HUGGINGFACEHUB_API_TOKEN not found, some features may not work")
        
    self.coordinator = AgentWorkflow(
        agents=[external_knowledge_agent, code_agent],
        root_agent="external_knowledge_agent")    
    
    def download_gaia_file(self, task_id: str, api_url: str = "https://agents-course-unit4-scoring.hf.space") -> str:
        """Download file associated with task_id"""
        try:
            response = requests.get(f"{api_url}/files/{task_id}", timeout=30)
            response.raise_for_status()
            
            filename = f"task_{task_id}_file"
            with open(filename, 'wb') as f:
                f.write(response.content)
            return filename
        except Exception as e:
            print(f"Failed to download file for task {task_id}: {e}")
            return None
    
    async def solve_gaia_question(self, question_data: Dict[str, Any]) -> str:
        """
        Solve GAIA question with enhanced validation and reformatting
        """
        question = question_data.get("Question", "")
        task_id = question_data.get("task_id", "")
        
        # Try to download file if task_id provided
        file_path = None
        if task_id:
            try:
                file_path = self.download_gaia_file(task_id)
                if file_path:
                    documents = read_and_parse_content(file_path)
            except Exception as e:
                print(f"Failed to download/process file for task {task_id}: {e}")
        
        # Prepare context prompt
        context_prompt = f"""
GAIA Task ID: {task_id}
Question: {question}
{f'File available: {file_path}' if file_path else 'No additional files'}
You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.""",        
        try:
            ctx = Context(self.coordinator)
            print("=== AGENT REASONING STEPS ===")
            
            handler = self.coordinator.run(ctx=ctx, user_msg=context_prompt)
            
            full_response = ""
            async for event in handler.stream_events():
                if isinstance(event, AgentStream):
                    print(event.delta, end="", flush=True)
                    full_response += event.delta
            
            final_response = await handler
            print("\n=== END REASONING ===")
            
            # Extract the final formatted answer
            final_answer = str(final_response).strip()
            
            print(f"Final GAIA formatted answer: {final_answer}")
            return final_answer
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(error_msg)
            return error_msg
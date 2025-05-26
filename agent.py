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

information_retrieval_tool = FunctionTool.from_defaults(
    fn=create_rag_tool_fn,
    name="information_retrieval_tool",
    description=(
        "This is the BEST and OPTIMAL tool to query information from documents parsed from URLs or files. "
        "Use this tool to build a Retrieval Augmented Generation (RAG) engine from documents AND optionally query it immediately. "
        "Input: documents (list of documents) and optional query parameter. "
        "If no query is provided: creates and returns a RAG query engine tool for later use. "
        "If query is provided: creates the RAG engine AND immediately returns the answer to your question. "
        "ALWAYS use this tool when you need to retrieve specific information from documents obtained via URLs or file. "
        "This dual-mode tool enables both RAG engine creation and direct question-answering in one step, making it the most efficient approach for document-based information retrieval."
    )
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

# 3. Create the final, customized FunctionTool for the agent.
# This is the tool you will actually give to your agent.
extract_url_tool = FunctionTool.from_defaults(
    fn=search_and_extract_top_url,
    name="extract_url_tool",
    description=(
        "Use this tool when you need to find a relevant URL to answer a question. It takes a search query as input and returns a single, relevant URL."
    )
)

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
        
    # Time Series
    "pytz": __import__("pytz"),
    
    # Utilities
    "tqdm": __import__("tqdm"),
    "pickle": __import__("pickle"),
    "gzip": __import__("gzip"),
    "base64": __import__("base64"),
    "hashlib": __import__("hashlib"),
    
    # Scientific Computing
    "sympy": __import__("sympy"),

    # llama-index
    "llama-index" : __import__("llama_index")
}

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
description="Execute Python code safely for calculations and data processing"
)

import re
from llama_index.core.tools import FunctionTool
from llama_index.llms.huggingface import HuggingFaceLLM

# --- 1. Initialize a dedicated LLM for Code Generation ---
# It's good practice to use a model specifically fine-tuned for coding.
# This model is loaded only once for efficiency.
code_llm = HuggingFaceLLM(
    model_name="Qwen/Qwen2.5-Coder-3B",
    tokenizer_name="Qwen/Qwen2.5-Coder-3B",
    device_map="auto",
    model_kwargs={"torch_dtype": "auto"},
    # Set generation parameters for precise, non-creative code output
    generate_kwargs={"temperature": 0.0, "do_sample": False}
)

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

IMPORTANT LIMITATIONS:
Your code will be executed in a restricted environment with limited functions and modules.
{str(safe_globals)}
Only use the functions and modules listed above. Do not use imports or other built-in functions.

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
        "Use this tool to generate executable Python code ONLY for mathematical calculations and problem solving. "
        "This tool is specifically designed for numerical computations, statistical analysis, algebraic operations, "
        "mathematical modeling, and scientific calculations."
        "DO NOT use this tool for document processing, text manipulation, or data parsing - use appropriate specialized tools instead."
        "The tool returns a string containing only the Python code for mathematical operations."
    )
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

# Create the simplified final answer tool
final_answer_function_tool = FunctionTool.from_defaults(
    fn=final_answer_tool,
    name="final_answer_tool",
    description=(
        "Use this tool to format the final answer according to GAIA requirements. "
        "Input the agent's response and the original question to get properly formatted output."
    )
)

class EnhancedGAIAAgent:
    def __init__(self):
        print("Initializing Enhanced GAIA Agent...")
        
        # Vérification du token HuggingFace
        hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not hf_token:
            print("Warning: HUGGINGFACEHUB_API_TOKEN not found, some features may not work")
        
        # Initialize only the tools that are actually defined in the file
        self.available_tools = [
            extract_url_tool,
            read_and_parse_tool,
            information_retrieval_tool,
            code_execution_tool,
            generate_code_tool,
        ]
                
        # Create main coordinator using only defined tools
        self.coordinator = ReActAgent(
            name="GAIACoordinator",
            system_prompt="""
You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
""",
            llm=proj_llm,
            tools=self.available_tools,
            max_steps=15,
            verbose=True,
            callback_manager=callback_manager,
        )
    
    
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
"""
        
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
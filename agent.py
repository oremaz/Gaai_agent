# Standard library imports
import logging
import os
import re
from typing import Dict, Any, List
from urllib.parse import urlparse
import torch

# Third-party imports
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer

# LlamaIndex core imports
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.agent.workflow import FunctionAgent, ReActAgent, AgentStream
from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser, UnstructuredElementNodeParser
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context
from llama_index.postprocessor.colpali_rerank import ColPaliRerank
from llama_index.core.schema import ImageNode, TextNode


# LlamaIndex specialized imports
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.readers.assemblyai import AssemblyAIAudioTranscriptReader
from llama_index.readers.json import JSONReader
from llama_index.readers.web import BeautifulSoupWebReader
from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
from llama_index.tools.arxiv import ArxivToolSpec
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.core.agent.workflow import AgentWorkflow

# --- Import all required official LlamaIndex Readers ---
from llama_index.readers.file import (
    PDFReader,
    DocxReader,
    CSVReader,
    PandasExcelReader)
from typing import List, Union
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.tools import QueryEngineTool
from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.query_pipeline import QueryPipeline

import importlib.util
import sys

import weave
weave.init("gaia-llamaindex-agents")
def get_max_memory_config(max_memory_per_gpu):
    """Generate max_memory config for available GPUs"""
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        max_memory = {}
        for i in range(num_gpus):
            max_memory[i] = max_memory_per_gpu
        return max_memory
    return None

model_id = "Qwen/Qwen3-14B-FP8"
proj_llm = HuggingFaceLLM(
    model_name=model_id,
    tokenizer_name=model_id,
    device_map="auto",
    max_new_tokens = 16000,
    model_kwargs={"torch_dtype": "auto"},
    generate_kwargs={
        "temperature": 0.6,
        "top_p": 0.95, 
        "top_k": 20
    }
)

code_llm = HuggingFaceLLM(
    model_name="Qwen/Qwen2.5-Coder-3B-Instruct",
    tokenizer_name="Qwen/Qwen2.5-Coder-3B-Instruct",
    device_map= "auto",      
    model_kwargs={
        "torch_dtype": "auto"},
    # Set generation parameters for precise, non-creative code output
    generate_kwargs={"do_sample": False}
)

embed_model = HuggingFaceEmbedding(
    model_name="llamaindex/vdr-2b-multi-v1",
    device="cpu",
    trust_remote_code=True,
    model_kwargs={
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True
    }
)

logging.basicConfig(level=logging.INFO)
logging.getLogger("llama_index.core.agent").setLevel(logging.DEBUG)
logging.getLogger("llama_index.llms").setLevel(logging.DEBUG)

Settings.llm = proj_llm
Settings.embed_model = embed_model

def read_and_parse_content(input_path: str) -> List[Document]:
    """
    Reads and parses content from a local file path into Document objects.
    URL handling has been moved to search_and_extract_top_url.
    """
    # Remove URL handling - this will now only handle local files
    if not os.path.exists(input_path):
        return [Document(text=f"Error: File not found at {input_path}")]

    file_extension = os.path.splitext(input_path)[1].lower()
    
    # Readers map
    readers_map = {
        '.pdf': PDFReader(),
        '.docx': DocxReader(),
        '.doc': DocxReader(),
        '.csv': CSVReader(),
        '.json': JSONReader(),
        '.xlsx': PandasExcelReader(),
    }

    if file_extension in ['.mp3', '.mp4', '.wav', '.m4a', '.flac']:
        try:
            loader = AssemblyAIAudioTranscriptReader(file_path=input_path)
            documents = loader.load_data()
            return documents
        except Exception as e:
            return [Document(text=f"Error transcribing audio: {e}")]

    if file_extension in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
        # Load the actual image content, not just the path
        try:
            with open(input_path, 'rb') as f:
                image_data = f.read()
            return [Document(
                text=f"IMAGE_CONTENT_BINARY",
                metadata={
                    "source": input_path, 
                    "type": "image", 
                    "path": input_path,
                    "image_data": image_data  # Store actual image data
                }
            )]
        except Exception as e:
            return [Document(text=f"Error reading image: {e}")]

    if file_extension in readers_map:
        loader = readers_map[file_extension]
        documents = loader.load_data(file=input_path)
    else:
        # Fallback for text files
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
            documents = [Document(text=content, metadata={"source": input_path})]
        except Exception as e:
            return [Document(text=f"Error reading file as plain text: {e}")]

    # Add source metadata
    for doc in documents:
        doc.metadata["source"] = input_path

    return documents

class DynamicQueryEngineManager:
    """Single unified manager for all RAG operations - replaces the entire static approach."""
    
    def __init__(self, initial_documents: List[str] = None):
        self.documents = []
        self.query_engine_tool = None
        
        # Load initial documents if provided
        if initial_documents:
            self._load_initial_documents(initial_documents)
        
        self._create_rag_tool()
    
    def _load_initial_documents(self, document_paths: List[str]):
        """Load initial documents using read_and_parse_content."""
        for path in document_paths:
            docs = read_and_parse_content(path)
            self.documents.extend(docs)
        print(f"Loaded {len(self.documents)} initial documents")
    
    def _create_rag_tool(self):
        """Create RAG tool using multimodal-aware parsing."""
        documents = self.documents if self.documents else [
            Document(text="No documents loaded yet. Use web search to add content.")
        ]
        
        # Separate text and image documents for proper processing
        text_documents = []
        image_documents = []
        
        for doc in documents:
            doc_type = doc.metadata.get("type", "")
            source = doc.metadata.get("source", "").lower()
            file_type = doc.metadata.get("file_type", "")
            
            # Identify image documents
            if (doc_type in ["image", "web_image"] or 
                file_type in ['jpg', 'png', 'jpeg', 'gif', 'bmp', 'webp'] or
                any(ext in source for ext in ['.jpg', '.png', '.jpeg', '.gif', '.bmp', '.webp'])):
                image_documents.append(doc)
            else:
                text_documents.append(doc)
        
        # Use UnstructuredElementNodeParser for text content with multimodal awareness
        element_parser = UnstructuredElementNodeParser()
        
        nodes = []
        
        # Process text documents with UnstructuredElementNodeParser
        if text_documents:
            try:
                text_nodes = element_parser.get_nodes_from_documents(text_documents)
                nodes.extend(text_nodes)
            except Exception as e:
                print(f"Error parsing text documents with UnstructuredElementNodeParser: {e}")
                # Fallback to simple parsing if UnstructuredElementNodeParser fails
                from llama_index.core.node_parser import SimpleNodeParser
                simple_parser = SimpleNodeParser.from_defaults(chunk_size=1024, chunk_overlap=200)
                text_nodes = simple_parser.get_nodes_from_documents(text_documents)
                nodes.extend(text_nodes)
        
        # Process image documents as ImageNodes
        if image_documents:
            for img_doc in image_documents:
                try:
                    image_node = ImageNode(
                        text=img_doc.text or f"Image content from {img_doc.metadata.get('source', 'unknown')}",
                        metadata=img_doc.metadata,
                        image_path=img_doc.metadata.get("path"),
                        image=img_doc.metadata.get("image_data")
                    )
                    nodes.append(image_node)
                except Exception as e:
                    print(f"Error creating ImageNode: {e}")
                    # Fallback to regular TextNode for images
                    text_node = TextNode(
                        text=img_doc.text or f"Image content from {img_doc.metadata.get('source', 'unknown')}",
                        metadata=img_doc.metadata
                    )
                    nodes.append(text_node)
        
        index = VectorStoreIndex(nodes)
        class HybridReranker:
            def __init__(self):
                self.text_reranker = SentenceTransformerRerank(
                    model="cross-encoder/ms-marco-MiniLM-L-2-v2", 
                    top_n=3
                )
                self.visual_reranker = ColPaliRerank(
                    top_n=3,
                    model="vidore/colpali-v1.2",
                    keep_retrieval_score=True,
                    device="cpu"
                )
            def postprocess_nodes(self, nodes, query_bundle):
                # Your exact implementation
                text_nodes = []
                visual_nodes = []
                
                for node in nodes:
                    if (hasattr(node, 'image_path') and node.image_path) or \
                       (hasattr(node, 'metadata') and node.metadata.get('file_type') in ['jpg', 'png', 'jpeg', 'pdf']) or \
                       (hasattr(node, 'metadata') and node.metadata.get('type') in ['image', 'web_image']):
                        visual_nodes.append(node)
                    else:
                        text_nodes.append(node)
                
                reranked_text = []
                reranked_visual = []
                
                if text_nodes:
                    reranked_text = self.text_reranker.postprocess_nodes(text_nodes, query_bundle)
                
                if visual_nodes:
                    reranked_visual = self.visual_reranker.postprocess_nodes(visual_nodes, query_bundle)
                
                combined_results = []
                max_len = max(len(reranked_text), len(reranked_visual))
                
                for i in range(max_len):
                    if i < len(reranked_text):
                        combined_results.append(reranked_text[i])
                    if i < len(reranked_visual):
                        combined_results.append(reranked_visual[i])
                
                return combined_results[:5]
        
        hybrid_reranker = HybridReranker()
        
        query_engine = index.as_query_engine(
            similarity_top_k=20,
            node_postprocessors=[hybrid_reranker],
            response_mode="tree_summarize"
        )
        
        self.query_engine_tool = QueryEngineTool.from_defaults(
            query_engine=query_engine,
            name="dynamic_hybrid_multimodal_rag_tool",
            description=(
                "Advanced dynamic knowledge base with hybrid reranking. "
                "Uses ColPali for visual content and SentenceTransformer for text content. "
                "Automatically updated with web search content."
            )
        )
    
    def add_documents(self, new_documents: List[Document]):
        """Add documents from web search and recreate tool."""
        self.documents.extend(new_documents)
        self._create_rag_tool()  # Recreate with ALL documents
        print(f"Added {len(new_documents)} documents. Total: {len(self.documents)}")
    
    def get_tool(self):
        return self.query_engine_tool
 
# Global instance
dynamic_qe_manager = DynamicQueryEngineManager()

# 1. Create the base DuckDuckGo search tool from the official spec.
# This tool returns text summaries of search results, not just URLs.
base_duckduckgo_tool = DuckDuckGoSearchToolSpec().to_tool_list()[1]

def search_and_extract_content_from_url(query: str) -> List[Document]:
    """
    Searches web, gets top URL, and extracts both text content and images.
    Returns a list of Document objects containing the extracted content.
    """
    # Get URL from search
    search_results = base_duckduckgo_tool(query, max_results=1)
    url_match = re.search(r"https?://\S+", str(search_results))
    
    if not url_match:
        return [Document(text="No URL could be extracted from the search results.")]
    
    url = url_match.group(0)[:-2]
    print(url)
    documents = []
    
    try:
        # Check if it's a YouTube URL
        if "youtube" in urlparse(url).netloc or "youtu.be" in urlparse(url).netloc:
            loader = YoutubeTranscriptReader()
            documents = loader.load_data(youtubelinks=[url])
        else:
            loader = BeautifulSoupWebReader()
            documents = loader.load_data(urls=[url])
        for doc in documents:
            doc.metadata["source"] = url
            doc.metadata["type"] = "web_text"
        return documents
    except Exception as e:
        # Handle any exceptions that occur during content extraction
        return [Document(text=f"Error extracting content from URL: {str(e)}")]
    
def enhanced_web_search_and_update(query: str) -> str:
    """
    Performs web search, extracts content, and adds it to the dynamic query engine.
    """
    # Extract content from web search
    documents = search_and_extract_content_from_url(query)
    
    # Add documents to the dynamic query engine
    if documents and not any("Error" in doc.text for doc in documents):
        dynamic_qe_manager.add_documents(documents)
        
        # Return summary of what was added
        text_docs = [doc for doc in documents if doc.metadata.get("type") == "web_text"]
        image_docs = [doc for doc in documents if doc.metadata.get("type") == "web_image"]
        
        summary = f"Successfully added web content to knowledge base:\n"
        summary += f"- {len(text_docs)} text documents\n"
        summary += f"- {len(image_docs)} images\n"
        summary += f"Source: {documents[0].metadata.get('source', 'Unknown')}"
        
        return summary
    else:
        error_msg = documents[0].text if documents else "No content extracted"
        return f"Failed to extract web content: {error_msg}"

# Create the enhanced web search tool
enhanced_web_search_tool = FunctionTool.from_defaults(
    fn=enhanced_web_search_and_update,
    name="enhanced_web_search",
    description="Search the web, extract content and images, and add them to the knowledge base for future queries."
)

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
        
        # Initialize the dynamic query engine manager
        self.dynamic_qe_manager = DynamicQueryEngineManager()
        
        # Create enhanced agents with dynamic tools
        self.external_knowledge_agent = ReActAgent(
            name="external_knowledge_agent", 
            description="Advanced information retrieval with dynamic knowledge base",
            system_prompt="""You are an advanced information specialist with a sophisticated RAG system. 
            Your knowledge base uses hybrid reranking and grows dynamically with each web search and document addition.
            Always add relevant content to your knowledge base, then query it for answers.""",
            tools=[
                enhanced_web_search_tool,
                self.dynamic_qe_manager.get_tool(),
                code_execution_tool
            ],
            llm=proj_llm,
            max_steps=8,
            verbose=True)
        
        self.code_agent = ReActAgent(
            name="code_agent",
            description="Handles Python code for calculations and data processing",
            system_prompt="You are a Python programming specialist. You work with Python code to perform calculations, data analysis, and mathematical operations.",
            tools=[code_execution_tool],
            llm=code_llm,
            max_steps=6,
            verbose=True)
        
        # Fixed indentation: coordinator initialization inside __init__
        self.coordinator = AgentWorkflow(
            agents=[self.external_knowledge_agent, self.code_agent],
            root_agent="external_knowledge_agent"
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
    
    def add_documents_to_knowledge_base(self, file_path: str):
        """Add downloaded GAIA documents to the dynamic knowledge base"""
        try:
            documents = read_and_parse_content(file_path)
            if documents:
                self.dynamic_qe_manager.add_documents(documents)
                print(f"Added {len(documents)} documents from {file_path} to dynamic knowledge base")
                
                # Update the agent's tools with the refreshed query engine
                self.external_knowledge_agent.tools = [
                    enhanced_web_search_tool,
                    self.dynamic_qe_manager.get_tool(),  # Get the updated tool
                    code_execution_tool
                ]
                return True
        except Exception as e:
            print(f"Failed to add documents from {file_path}: {e}")
            return False
    
    async def solve_gaia_question(self, question_data: Dict[str, Any]) -> str:
        """
        Solve GAIA question with dynamic knowledge base integration
        """
        question = question_data.get("Question", "")
        task_id = question_data.get("task_id", "")
        
        # Try to download and add file to knowledge base if task_id provided
        file_path = None
        if task_id:
            try:
                file_path = self.download_gaia_file(task_id)
                if file_path:
                    # Add documents to dynamic knowledge base
                    self.add_documents_to_knowledge_base(file_path)
                    print(f"Successfully integrated GAIA file into dynamic knowledge base")
            except Exception as e:
                print(f"Failed to download/process file for task {task_id}: {e}")
        
        # Enhanced context prompt with dynamic knowledge base awareness
        context_prompt = f"""
GAIA Task ID: {task_id}
Question: {question}
{f'File processed and added to knowledge base: {file_path}' if file_path else 'No additional files'}

You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."""
        
        try:
            ctx = Context(self.coordinator)
            print("=== AGENT REASONING STEPS ===")
            print(f"Dynamic knowledge base contains {len(self.dynamic_qe_manager.documents)} documents")
            
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
            print(f"Knowledge base now contains {len(self.dynamic_qe_manager.documents)} documents")
            
            return final_answer
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(error_msg)
            return error_msg
    
    def get_knowledge_base_stats(self):
        """Get statistics about the current knowledge base"""
        return {
            "total_documents": len(self.dynamic_qe_manager.documents),
            "document_sources": [doc.metadata.get("source", "Unknown") for doc in self.dynamic_qe_manager.documents]
        }

import asyncio

async def main():
    agent = EnhancedGAIAAgent()
    question_data = {
        "Question": "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia.",
        "task_id": ""
    }
    print(question_data)
    answer = await agent.solve_gaia_question(question_data)
    print(f"Answer: {answer}")

if __name__ == '__main__':
    asyncio.run(main())
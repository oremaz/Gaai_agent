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

llama_debug = LlamaDebugHandler(print_trace_on_end=True)
callback_manager = CallbackManager([llama_debug])


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


class EnhancedRAGQueryEngine:
    def __init__(self, task_context: str = ""):
        self.task_context = task_context
        self.embed_model = embed_model
        self.reranker = SentenceTransformerRerank(model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=5)
        
        self.readers = {
            '.pdf': PDFReader(),
            '.docx': DocxReader(),
            '.doc': DocxReader(),
            '.csv': CSVReader(),
            '.txt': lambda file_path: [Document(text=open(file_path, 'r').read())],
            '.jpg': ImageReader(),
            '.jpeg': ImageReader(),
            '.png': ImageReader()
        }
        
        self.sentence_window_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text"
        )
        
        self.hierarchical_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[2048, 512, 128]
        )
    
    def load_and_process_documents(self, file_paths: List[str]) -> List[Document]:
        documents = []
        
        for file_path in file_paths:
            file_ext = os.path.splitext(file_path)[1].lower()
            
            try:
                if file_ext in self.readers:
                    reader = self.readers[file_ext]
                    if callable(reader):
                        docs = reader(file_path)
                    else:
                        docs = reader.load_data(file=file_path)
                    
                    # Add metadata to all documents
                    for doc in docs:
                        doc.metadata.update({
                            "file_path": file_path,
                            "file_type": file_ext[1:],
                            "task_context": self.task_context
                        })
                    documents.extend(docs)
                        
            except Exception as e:
                # Fallback to text reading
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    documents.append(Document(
                        text=content,
                        metadata={"file_path": file_path, "file_type": "text", "error": str(e)}
                    ))
                except:
                    print(f"Failed to process {file_path}: {e}")
        
        return documents
    
    def create_advanced_index(self, documents: List[Document], use_hierarchical: bool = False) -> VectorStoreIndex:
        if use_hierarchical or len(documents) > 10:
            nodes = self.hierarchical_parser.get_nodes_from_documents(documents)
        else:
            nodes = self.sentence_window_parser.get_nodes_from_documents(documents)
        
        index = VectorStoreIndex(
            nodes,
            embed_model=self.embed_model
        )
        
        return index
    
    def create_context_aware_query_engine(self, index: VectorStoreIndex):
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=10,
            embed_model=self.embed_model
        )
        
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[self.reranker],
            llm=proj_llm
        )
        
        return query_engine

def comprehensive_rag_analysis(file_paths: List[str], query: str, task_context: str = "") -> str:
    try:
        rag_engine = EnhancedRAGQueryEngine(task_context)
        documents = rag_engine.load_and_process_documents(file_paths)
        
        if not documents:
            return "No documents could be processed successfully."
        
        total_text_length = sum(len(doc.text) for doc in documents)
        use_hierarchical = total_text_length > 50000 or len(documents) > 5
        
        index = rag_engine.create_advanced_index(documents, use_hierarchical)
        query_engine = rag_engine.create_context_aware_query_engine(index)
        
        enhanced_query = f"""
        Task Context: {task_context}
        Original Query: {query}
        
        Please analyze the provided documents and answer the query with precise, factual information.
        """
        
        response = query_engine.query(enhanced_query)
        
        result = f"**RAG Analysis Results:**\n\n"
        result += f"**Documents Processed:** {len(documents)}\n"
        result += f"**Answer:**\n{response.response}\n\n"
        
        return result
        
    except Exception as e:
        return f"RAG analysis failed: {str(e)}"

def cross_document_analysis(file_paths: List[str], query: str, task_context: str = "") -> str:
    try:
        rag_engine = EnhancedRAGQueryEngine(task_context)
        all_documents = []
        document_groups = {}
        
        for file_path in file_paths:
            docs = rag_engine.load_and_process_documents([file_path])
            doc_key = os.path.basename(file_path)
            document_groups[doc_key] = docs
            
            for doc in docs:
                doc.metadata.update({
                    "document_group": doc_key,
                    "total_documents": len(file_paths)
                })
            all_documents.extend(docs)
        
        index = rag_engine.create_advanced_index(all_documents, use_hierarchical=True)
        query_engine = rag_engine.create_context_aware_query_engine(index)
        
        response = query_engine.query(f"Task: {task_context}\nQuery: {query}")
        
        result = f"**Cross-Document Analysis:**\n"
        result += f"**Documents:** {list(document_groups.keys())}\n"
        result += f"**Answer:**\n{response.response}\n"
        
        return result
        
    except Exception as e:
        return f"Cross-document analysis failed: {str(e)}"

# Create tools
enhanced_rag_tool = FunctionTool.from_defaults(
    fn=comprehensive_rag_analysis,
    name="Enhanced RAG Analysis",
    description="Comprehensive document analysis using advanced RAG with hybrid search and context-aware processing"
)

cross_document_tool = FunctionTool.from_defaults(
    fn=cross_document_analysis,
    name="Cross-Document Analysis", 
    description="Advanced analysis across multiple documents with cross-referencing capabilities"
)

# Analysis Agent
analysis_agent = FunctionAgent(
    name="AnalysisAgent",
    description="Advanced multimodal analysis using enhanced RAG with hybrid search and cross-document capabilities",
    system_prompt="""
    You are an advanced analysis specialist with access to:
    - Enhanced RAG with hybrid search and reranking
    - Multi-format document processing (PDF, Word, CSV, images, text)
    - Cross-document analysis and synthesis
    - Context-aware query processing
    
    Your capabilities:
    1. Process multiple file types simultaneously
    2. Perform semantic search across document collections
    3. Cross-reference information between documents
    4. Extract precise information with source attribution
    5. Handle both text and visual content analysis
    
    Always consider the GAIA task context and provide precise, well-sourced answers.
    """,
    llm=proj_llm,
    tools=[enhanced_rag_tool, cross_document_tool],
    max_steps=5, 
    verbose = True
)

class IntelligentSourceRouter:
    def __init__(self):
        # Initialize ArXiv and DuckDuckGo as LlamaIndex tools
        self.arxiv_tool = ArxivToolSpec().to_tool_list()[0]
        self.duckduckgo_tool = DuckDuckGoSearchToolSpec().to_tool_list()[1]

    def detect_intent_and_extract_content(self, query: str, max_results = 1) -> str:
        # Use your LLM to decide between arxiv and web_search
        intent_prompt = f"""
        Analyze this query and determine if it's scientific research or general information:
        Query: "{query}"
        Choose ONE source:
        - arxiv: For scientific research, academic papers, technical studies, algorithms, experiments
        - web_search: For all other information (current events, general facts, weather, how-to guides, etc.)
        Respond with ONLY "arxiv" or "web_search".
        """
        response = proj_llm.complete(intent_prompt)
        selected_source = response.text.strip().lower()

        results = [f"**Query**: {query}", f"**Selected Source**: {selected_source}", "="*50]
        try:
            if selected_source == 'arxiv':
                result = self.arxiv_tool.call(query=query)
                results.append(f"**ArXiv Research:**\n{result}")
            else:
                result = self.duckduckgo_tool.call(query=query, max_results=max_results)
                # Format results if needed
                if isinstance(result, list):
                    formatted = []
                    for i, r in enumerate(result, 1):
                        formatted.append(
                            f"{i}. **{r.get('title', '')}**\n   URL: {r.get('href', '')}\n   {r.get('body', '')}"
                        )
                    result = "\n".join(formatted)
                results.append(f"**Web Search Results:**\n{result}")
                print (results)
        except Exception as e:
            results.append(f"**Search failed**: {str(e)}")
        return "\n\n".join(results)

# Initialize router
intelligent_router = IntelligentSourceRouter()

# Create enhanced research tool
def enhanced_smart_research_tool(query: str, task_context: str = "") -> str:
    full_query = f"{query} {task_context}".strip()
    return intelligent_router.detect_intent_and_extract_content(full_query)

research_tool = FunctionTool.from_defaults(
    fn=enhanced_smart_research_tool,
    name="Research Tool",
    description="""Intelligent research specialist that automatically routes between scientific and general sources and extract content. Use this tool at least when you need:
    
    **Scientific Research (ArXiv + Content Extraction):**
    
    **General Research (Web + Content Extraction):**
    
    **Automatic Features:**
    - Intelligently selects between ArXiv and web search
    - Extracts full content from web pages (not just snippets)
    - Provides source attribution and detailed information
    
    **When to use:** Questions requiring external knowledge not in your training data, current events, scientific research, or factual verification.
    
    **Input format:** Provide the research query with any relevant context."""
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

# Code Agent as ReActAgent with explicit code generation
code_agent = ReActAgent(
    name="CodeAgent",
    description="Advanced calculations, data processing using code generation and execution",
    system_prompt="""
    You are a coding specialist. For EVERY computational task:
    
    1. THINK: Analyze what calculation/processing is needed
    2. GENERATE CODE: Write Python code to solve the problem
    3. EXECUTE: Use the Python Code Execution tool to run your code
    4. OBSERVE: Check the results
    5. REPEAT if needed
    
    ALWAYS write code for:
    - Mathematical calculations
    - Data processing
    - Numerical analysis
    - Text processing
    - Any computational task
    
    Example workflow:
    Question: "What is 15 * 23 + 7?"
    
    Thought: I need to calculate 15 * 23 + 7
    Action: Python Code Execution
    Action Input: {"code": "result = 15 * 23 + 7\nprint(f'The answer is: {result}')"}
    
    Store your final answer in a variable called 'result'.
    """,
    llm=proj_llm,
    tools=[code_execution_tool],
    max_steps=5,
    verbose=True, 
    callback_manager=callback_manager,
)

def analysis_function(query: str, files=None):
    ctx = Context(analysis_agent)
    return analysis_agent.run(query, ctx=ctx)

def code_function(query: str):
    ctx = Context(code_agent)
    return code_agent.run(query, ctx=ctx)


analysis_tool = FunctionTool.from_defaults(
    fn=analysis_function,
    name="AnalysisAgent",
    description="""Advanced multimodal document analysis specialist. Use this tool at least when you need to:
    
    **Document Processing:**
    - Analyze PDF, Word, CSV, or image files provided with the question
    - Extract specific information from tables, charts, or structured documents
    - Cross-reference information across multiple documents
    - Perform semantic search within document collections
    
    **Content Analysis:**
    - Summarize long documents or extract key facts
    - Find specific data points, numbers, or text within files
    - Analyze visual content in images (charts, graphs, diagrams)
    - Compare information between different document sources
    
    **When to use:** Questions involving file attachments, document analysis, data extraction from PDFs/images, or when you need to process structured/unstructured content.
    
    **Input format:** Provide the query and mention any relevant files or context."""
)

code_tool = FunctionTool.from_defaults(
    fn=code_function,
    name="CodeAgent",
    description="""Advanced computational specialist using ReAct reasoning. Use this tool at least when you need:
    
    **Core Capabilities:**
    - **Autonomous Code Generation**: Writes Python code from scratch to solve computational problems
    - **Multi-step Problem Solving**: Breaks complex tasks into manageable coding steps
    - **Self-debugging**: Identifies and fixes errors through iterative refinement
    - **Library Integration**: Leverages numpy, pandas, matplotlib, scipy, sklearn, and other scientific libraries
    - **Result Verification**: Validates outputs and adjusts approach as needed
    
    **When to Use:**
    - Mathematical calculations requiring step-by-step computation
    - Data analysis and statistical processing
    - Algorithm implementation and optimization
    - Numerical simulations and modeling
    - Text processing and pattern analysis
    - Complex logical operations requiring code verification
    
    **Unique Advantage**: Unlike simple calculation tools, this agent can autonomously write, execute, debug, and refine code until achieving the correct solution, making it ideal for complex computational tasks that require adaptive problem-solving.
    
    **Input Format**: Describe the computational task clearly, including any data, constraints, or specific requirements."""
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
            
            **1. AnalysisAgent** - Advanced multimodal document analysis specialist
            - Use for: PDF, Word, CSV, image file analysis
            - Capabilities: Extract data from tables/charts, cross-reference documents, semantic search
            - When to use: Questions with file attachments, document analysis, data extraction
            
            **2. Research Tool** - Intelligent research specialist with automatic routing
            - Use for: External knowledge, current events, scientific papers
            - Capabilities: Auto-routes between ArXiv (scientific) and web search (general), extracts full content
            - When to use: Questions requiring external knowledge, factual verification, current information
            
            **3. CodeAgent** - Advanced computational specialist using ReAct reasoning
            - Use for: Mathematical calculations, data processing, logical operations
            - Capabilities: Generates and executes Python code, handles complex computations, step-by-step problem solving
            - When to use: Precise calculations, data manipulation, mathematical problem solving
            
            IMPORTANT: Use tools strategically - only when their specific expertise is needed.
            For simple questions, you can answer directly without using any tools.
            
            CRITICAL: Your final answer must be EXACT and CONCISE as required by GAIA format:
            - For numbers: provide only the number (e.g., "42" or "3.14")
            - For strings: provide only the exact string (e.g., "Paris" or "Einstein")
            - For lists: use comma separation (e.g., "apple, banana, orange")
            - NO explanations, NO additional text, ONLY the precise answer
            """,
            llm=proj_llm,
            tools=[analysis_tool, research_tool, code_tool], 
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
        print("data",question_data)

        try:
            file_path = self.download_gaia_file(task_id)
        except FileNotFoundError as e:
            print(f"File not found for task {task_id}: {e}")
            file_path = None
        except Exception as e:
            print(f"Unexpected error downloading file for task {task_id}: {e}")
            file_path = None

        context_prompt = f"""
        GAIA Task ID: {task_id}
        Question: {question}
        {'File downloaded: ' + file_path if file_path else 'No files referenced'}
        
        Analyze this question and provide your reasoning and final answer.
        """
        
        try:
            from llama_index.core.workflow import Context
            ctx = Context(self.coordinator)
            raw_response = await self.coordinator.run(ctx=ctx, user_msg=context_prompt)
            
            # Post-process to extract exact GAIA format
            formatted_answer = await self.format_gaia_answer(str(raw_response), question)
            
            print(f"Formatted answer: {formatted_answer}")
            
            return formatted_answer
            
        except Exception as e:
            return f"Error processing question: {str(e)}"

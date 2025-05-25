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
from llama_index.core.tools.ondemand_loader_tool import OnDemandLoaderTool
from llama_index.tools.arxiv import ArxivToolSpec
import duckduckgo_search as ddg
import re
from llama_index.core.agent.workflow import ReActAgent
import wandb
from llama_index.callbacks.wandb import WandbCallbackHandler
from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.callbacks.llama_debug import LlamaDebugHandler
from llama_index.core import Settings
from llama_index.core.agent.workflow import CodeActAgent

from transformers import AutoModelForCausalLM, AutoTokenizer
from llama_index.llms.huggingface import HuggingFaceLLM

model_id = "Qwen/Qwen2.5-7B-Instruct" 
proj_llm = HuggingFaceLLM(
    model_name=model_id,
    tokenizer_name=model_id,
    device_map="auto",           # will use GPU if available
    model_kwargs={"torch_dtype": "auto"},
    generate_kwargs={"temperature": 0.7, "top_p": 0.95}
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
    max_steps=5
)


from llama_index.tools.arxiv import ArxivToolSpec
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec

class IntelligentSourceRouter:
    def __init__(self):
        # Initialize ArXiv and DuckDuckGo as LlamaIndex tools
        self.arxiv_tool = ArxivToolSpec().to_tool_list()[0]
        self.duckduckgo_tool = DuckDuckGoSearchToolSpec().to_tool_list()[0]

    def detect_intent_and_route(self, query: str) -> str:
        # Use your LLM to decide between arxiv and web_search
        intent_prompt = f"""
        Analyze this query and determine if it's scientific research or general information:
        Query: "{query}"
        Choose ONE source:
        - arxiv: For scientific research, academic papers, technical studies, algorithms, experiments
        - web_search: For all other information (current events, general facts, weather, how-to guides, etc.)
        Respond with ONLY "arxiv" or "web_search".
        """
        response = text_llm.complete(intent_prompt)
        selected_source = response.text.strip().lower()

        results = [f"**Query**: {query}", f"**Selected Source**: {selected_source}", "="*50]
        try:
            if selected_source == 'arxiv':
                result = self.arxiv_tool.call(query=query, max_results=3)
                results.append(f"**ArXiv Research:**\n{result}")
            else:
                result = self.duckduckgo_tool.call(query=query, max_results=5)
                # Format results if needed
                if isinstance(result, list):
                    formatted = []
                    for i, r in enumerate(result, 1):
                        formatted.append(
                            f"{i}. **{r.get('title', '')}**\n   URL: {r.get('href', '')}\n   {r.get('body', '')}"
                        )
                    result = "\n".join(formatted)
                results.append(f"**Web Search Results:**\n{result}")
        except Exception as e:
            results.append(f"**Search failed**: {str(e)}")
        return "\n\n".join(results)

class IntelligentSourceRouter:
    def __init__(self):
        # Initialize Arxiv and DuckDuckGo tools
        self.arxiv_tool = ArxivToolSpec().to_tool_list()[0]
        self.duckduckgo_tool = DuckDuckGoSearchToolSpec().to_tool_list()[0]

    def detect_intent_and_extract_content(self, query: str, max_results: int = 3) -> str:
        # Use your LLM to decide between arxiv and web_search
        intent_prompt = f"""
        Analyze this query and determine if it's scientific research or general information:
        Query: "{query}"
        Choose ONE source:
        - arxiv: For scientific research, academic papers, technical studies, algorithms, experiments
        - web_search: For all other information (current events, general facts, weather, how-to guides, etc.)
        Respond with ONLY "arxiv" or "web_search".
        """
        response = text_llm.complete(intent_prompt)
        selected_source = response.text.strip().lower()

        results = [f"**Query**: {query}", f"**Selected Source**: {selected_source}", "="*50]
        try:
            if selected_source == 'arxiv':
                # Extract abstracts and paper summaries (deep content)
                arxiv_results = self.arxiv_tool.call(query=query, max_results=max_results)
                results.append(f"**Extracted ArXiv Content:**\n{arxiv_results}")
            else:
                # DuckDuckGo returns a list of dicts with 'href', 'title', 'body'
                web_results = self.duckduckgo_tool.call(query=query, max_results=max_results)
                if isinstance(web_results, list):
                    formatted = []
                    for i, r in enumerate(web_results, 1):
                        formatted.append(
                            f"{i}. **{r.get('title', '')}**\n   URL: {r.get('href', '')}\n   {r.get('body', '')}"
                        )
                    web_content = "\n".join(formatted)
                else:
                    web_content = str(web_results)
                results.append(f"**Extracted Web Content:**\n{web_content}")
        except Exception as e:
            results.append(f"**Extraction failed**: {str(e)}")
        return "\n\n".join(results)

# Initialize router
intelligent_router = IntelligentSourceRouter()

# Create enhanced research tool
def enhanced_smart_research_tool(query: str, task_context: str = "", max_results: int = 3) -> str:
    full_query = f"{query} {task_context}".strip()
    return intelligent_router.detect_intent_and_extract_content(full_query, max_results=max_results)

research_tool = FunctionTool.from_defaults(
    fn=enhanced_smart_research_tool,
    name="Research Tool",
    description="""Intelligent research specialist that automatically routes between scientific and general sources. Use this tool when you need:
    
    **Scientific Research (ArXiv):**
    - Academic papers, research studies, technical algorithms
    - Scientific experiments, theories, mathematical concepts
    - Recent developments in AI, ML, physics, chemistry, etc.
    
    **General Research (Web + Content Extraction):**
    - Current events, news, real-time information
    - Biographical information, company details, locations
    - How-to guides, technical documentation
    - Weather data, sports results, cultural information
    - Product specifications, reviews, comparisons
    
    **Automatic Features:**
    - Intelligently selects between ArXiv and web search
    - Extracts full content from web pages (not just snippets)
    - Provides source attribution and detailed information
    
    **When to use:** Questions requiring external knowledge not in your training data, current events, scientific research, or factual verification.
    
    **Input format:** Provide the research query with any relevant context."""
)

code_agent = CodeActAgent(
    name="CodeAgent",
    description="Advanced calculations, data processing, and final answer synthesis using ReAct reasoning",
    system_prompt="""
    You are a coding and reasoning specialist using ReAct methodology.

    For each task:
    1. THINK: Analyze what needs to be calculated or processed
    2. ACT: Execute appropriate code or calculations
    3. OBSERVE: Review results and determine if more work is needed
    4. REPEAT: Continue until you have the final answer

    Always show your reasoning process clearly and provide exact answers as required by GAIA.
    """,
    llm=proj_llm,  # Your language model instance
    max_steps=5    # Optional: limit the number of reasoning steps
)

analysis_tool = FunctionTool.from_defaults(
    fn=analysis_function,
    name="AnalysisAgent",
    description="""Advanced multimodal document analysis specialist. Use this tool when you need to:
    
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
    description="""Advanced computational specialist using ReAct reasoning. Use this tool when you need:
    
    **Mathematical Calculations:**
    - Complex arithmetic, algebra, statistics, probability
    - Unit conversions, percentage calculations
    - Financial calculations (interest, loans, investments)
    - Scientific calculations (physics, chemistry formulas)
    
    **Data Processing:**
    - Parsing and analyzing numerical data
    - String manipulation and text processing
    - Date/time calculations and conversions
    - List operations, sorting, filtering
    
    **Logical Operations:**
    - Step-by-step problem solving with code
    - Verification of calculations or logic
    - Pattern analysis and data validation
    - Algorithm implementation for specific problems
    
    **Programming Tasks:**
    - Code generation for specific computational needs
    - Data structure manipulation
    - Regular expression operations
    
    **When to use:** Questions requiring precise calculations, data manipulation, logical reasoning with code, or when you need to verify numerical results.
    
    **Input format:** Describe the calculation or processing task clearly, including any specific requirements or constraints."""
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
            description="Main GAIA coordinator that uses specialist agents as intelligent tools",
            system_prompt="""
            You are the main GAIA coordinator using ReAct reasoning methodology.
            
            Your process:
            1. THINK: Analyze the GAIA question thoroughly            
            2. ACT: Use your specialist tools IF RELEVANT            
            3. OBSERVE: Review results from specialist tools 
            4. REPEAT: Continue until you have the final answer. If you give a final answer, FORMAT: Ensure answer is EXACT GAIA format (number only, word only, etc.)

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
            max_steps = 10
        )
    
    async def solve_gaia_question(self, question_data: Dict[str, Any]) -> str:
        question = question_data.get("Question", "")
        task_id = question_data.get("task_id", "")
        context_prompt = f"""
        GAIA Task ID: {task_id}
        Question: {question}
        {f"Associated files: {question_data.get('file_name', '')}" if 'file_name' in question_data else 'No files provided'}
        Instructions:
        1. Analyze this GAIA question using ReAct reasoning
        2. Use specialist tools ONLY when their specific expertise is needed
        3. Provide a precise, exact answer in GAIA format
        Begin your reasoning process:
        """
        try:
            from llama_index.core.workflow import Context
            ctx = Context(self.coordinator)
            response = await self.coordinator.run(ctx=ctx, user_msg=context_prompt)
            print (response)
            return str(response)
        except Exception as e:
            return f"Error processing question: {str(e)}"

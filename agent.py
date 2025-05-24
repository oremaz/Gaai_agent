from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool
from llama_index.core import VectorStoreIndex, Document
from llama_index.core.node_parser import SentenceWindowNodeParser, HierarchicalNodeParser
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.readers.file import PDFReader, DocxReader, CSVReader, ImageReader
import os
from typing import List, Dict, Any
from llama_index.readers.web import SimpleWebPageReader
from llama_index.core.tools.ondemand_loader_tool import OnDemandLoaderTool
from llama_index.tools.arxiv import ArxivToolSpec
import duckduckgo_search as ddg
import re
from llama_index.core.agent.workflow import ReActAgent
from llama_index.llms.openrouter import OpenRouter

text_llm = OpenRouter(
    model="mistralai/mistral-small-3.1-24b-instruct:free",  # as listed on OpenRouter
    api_key=os.getenv("OPENROUTER_API_KEY"),  # or pass your key directly
)
multimodal_llm = text_llm


class EnhancedRAGQueryEngine:
    def __init__(self, task_context: str = ""):
        self.task_context = task_context
        self.embed_model = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")
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
            llm=multimodal_llm
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
    llm=multimodal_llm,
    tools=[enhanced_rag_tool, cross_document_tool],
    max_steps=5,

)


class IntelligentSourceRouter:
    def __init__(self):
        # Initialize tools - only ArXiv and web search
        self.arxiv_spec = ArxivToolSpec()
        
        # Add web content loader
        self.web_reader = SimpleWebPageReader()
        
        # Create OnDemandLoaderTool for web content
        self.web_loader_tool = OnDemandLoaderTool.from_defaults(
            self.web_reader,
            name="Web Content Loader",
            description="Load and analyze web page content with intelligent chunking and search"
        )
            
    def web_search_fallback(self, query: str, max_results: int = 5) -> str:
        try:
            results = ddg.DDGS().text(query, max_results=max_results)
            return "\n".join([f"{i}. **{r['title']}**\n   URL: {r['href']}\n   {r['body']}" for i, r in enumerate(results, 1)])
        except Exception as e:
            return f"Search failed: {str(e)}"
    
    def extract_web_content(self, urls: List[str], query: str) -> str:
        """Extract and analyze content from web URLs"""
        try:
            content_results = []
            for url in urls[:3]:  # Limit to top 3 URLs
                try:
                    result = self.web_loader_tool.call(
                        urls=[url],
                        query=f"Extract information relevant to: {query}"
                    )
                    content_results.append(f"**Content from {url}:**\n{result}")
                except Exception as e:
                    content_results.append(f"**Failed to load {url}**: {str(e)}")
            
            return "\n\n".join(content_results)
        except Exception as e:
            return f"Content extraction failed: {str(e)}"
    
    def detect_intent_and_route(self, query: str) -> str:
        # Simple LLM-based discrimination: scientific vs non-scientific
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
        
        # Execute search and extract content
        results = [f"**Query**: {query}", f"**Selected Source**: {selected_source}", "="*50]
        
        try:
            if selected_source == 'arxiv':
                result = self.arxiv_spec.to_tool_list()[0].call(query=query, max_results=3)
                results.append(f"**ArXiv Research:**\n{result}")
                
            else:  # Default to web_search for everything else
                # Get search results
                search_results = self.web_search_fallback(query, 5)
                results.append(f"**Web Search Results:**\n{search_results}")
                
                # Extract URLs and load content
                urls = re.findall(r'URL: (https?://[^\s]+)', search_results)
                if urls:
                    web_content = self.extract_web_content(urls, query)
                    results.append(f"**Extracted Web Content:**\n{web_content}")
                    
        except Exception as e:
            results.append(f"**Search failed**: {str(e)}")
        
        return "\n\n".join(results)

# Initialize router
intelligent_router = IntelligentSourceRouter()

# Create enhanced research tool
def enhanced_smart_research_tool(query: str, task_context: str = "", max_results: int = 5) -> str:
    full_query = f"{query} {task_context}".strip()
    return intelligent_router.detect_intent_and_route(full_query)

research_tool = FunctionTool.from_defaults(
    fn=enhanced_smart_research_tool,
    name="Enhanced Research Tool",
    description="Intelligent research tool that discriminates between scientific (ArXiv) and general (web) research with deep content extraction"
)

def execute_python_code(code: str) -> str:
    try:
        safe_globals = {
            "__builtins__": {
                "len": len, "str": str, "int": int, "float": float,
                "list": list, "dict": dict, "sum": sum, "max": max, "min": min,
                "round": round, "abs": abs, "sorted": sorted
            },
            "math": __import__("math"),
            "datetime": __import__("datetime"),
            "re": __import__("re")
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

# Code Agent as ReActAgent
code_agent = ReActAgent(
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
    llm=text_llm,
    tools=[code_execution_tool],
    max_steps = 5
)

# Créer des outils à partir des agents
def analysis_function(query: str, files=None):
    ctx = Context(analysis_agent)
    return analysis_agent.run(query, ctx=ctx)


def code_function(query: str):
    ctx = Context(code_agent)
    return code_agent.run(query, ctx=ctx)

analysis_tool = FunctionTool.from_defaults(
    fn=analysis_function,
    name="AnalysisAgent",
    description="Advanced multimodal analysis using enhanced RAG"
)

code_tool = FunctionTool.from_defaults(
    fn=code_function,
    name="CodeAgent",
    description="Advanced calculations and data processing"
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
            llm=text_llm,
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
            response = await self.coordinator.run(ctx=ctx, input=context_prompt)
            return str(response)
        except Exception as e:
            return f"Error processing question: {str(e)}"

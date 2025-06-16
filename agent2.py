import os
import requests
import base64
from typing import Dict, Any, List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from smolagents import CodeAgent, OpenAIServerModel, tool, Tool
from smolagents.vision_web_browser import initialize_driver, save_screenshot, helium_instructions
from smolagents.agents import ActionStep
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import helium
from PIL import Image
from io import BytesIO
from time import sleep

# Langfuse observability imports
from opentelemetry.sdk.trace import TracerProvider
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry import trace
from opentelemetry.trace import format_trace_id
from langfuse import Langfuse


class BM25RetrieverTool(Tool):
    """
    BM25 retriever tool for document search when text documents are available
    """
    name = "bm25_retriever"
    description = "Uses BM25 search to retrieve relevant parts of uploaded documents. Use this when the question references an attached file or document."
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to find relevant document sections.",
        }
    }
    output_type = "string"

    def __init__(self, docs=None, **kwargs):
        super().__init__(**kwargs)
        self.docs = docs or []
        self.retriever = None
        if self.docs:
            self.retriever = BM25Retriever.from_documents(self.docs, k=5)

    def forward(self, query: str) -> str:
        if not self.retriever:
            return "No documents loaded for retrieval."

        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(query)
        return "\nRetrieved documents:\n" + "".join([
            f"\n\n===== Document {str(i)} =====\n" + doc.page_content
            for i, doc in enumerate(docs)
        ])


@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """Search for text on the current page via Ctrl + F and jump to the nth occurrence.
    
    Args:
        text: The text string to search for on the webpage
        nth_result: Which occurrence to jump to (default is 1 for first occurrence)
        
    Returns:
        str: Result of the search operation with match count and navigation status
    """
    try:
        driver = helium.get_driver()
        elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
        if nth_result > len(elements):
            return f"Match n°{nth_result} not found (only {len(elements)} matches found)"
        result = f"Found {len(elements)} matches for '{text}'."
        elem = elements[nth_result - 1]
        driver.execute_script("arguments[0].scrollIntoView(true);", elem)
        result += f"Focused on element {nth_result} of {len(elements)}"
        return result
    except Exception as e:
        return f"Error searching for text: {e}"


@tool
def go_back() -> str:
    """Navigate back to the previous page in browser history.
    
    Returns:
        str: Confirmation message or error description
    """
    try:
        driver = helium.get_driver()
        driver.back()
        return "Navigated back to previous page"
    except Exception as e:
        return f"Error going back: {e}"


@tool
def close_popups() -> str:
    """Close any visible modal or pop-up on the page by sending ESC key.
    
    Returns:
        str: Confirmation message or error description
    """
    try:
        driver = helium.get_driver()
        webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
        return "Attempted to close popups"
    except Exception as e:
        return f"Error closing popups: {e}"


@tool
def scroll_page(direction: str = "down", amount: int = 3) -> str:
    """Scroll the webpage in the specified direction.
    
    Args:
        direction: Direction to scroll, either 'up' or 'down'
        amount: Number of scroll actions to perform
        
    Returns:
        str: Confirmation message or error description
    """
    try:
        driver = helium.get_driver()
        for _ in range(amount):
            if direction.lower() == "down":
                driver.execute_script("window.scrollBy(0, 300);")
            elif direction.lower() == "up":
                driver.execute_script("window.scrollBy(0, -300);")
            sleep(0.5)
        return f"Scrolled {direction} {amount} times"
    except Exception as e:
        return f"Error scrolling: {e}"


@tool
def get_page_text() -> str:
    """Extract all visible text from the current webpage.
    
    Returns:
        str: The visible text content of the page
    """
    try:
        driver = helium.get_driver()
        text = driver.find_element(By.TAG_NAME, "body").text
        return f"Page text (first 2000 chars): {text[:2000]}"
    except Exception as e:
        return f"Error getting page text: {e}"


def save_screenshot_callback(memory_step: ActionStep, agent: CodeAgent) -> None:
    """Save screenshots for web browser automation"""
    try:
        sleep(1.0)
        driver = helium.get_driver()
        if driver is not None:
            # Clean up old screenshots
            for previous_memory_step in agent.memory.steps:
                if isinstance(previous_memory_step, ActionStep) and previous_memory_step.step_number <= memory_step.step_number - 2:
                    previous_memory_step.observations_images = None

            png_bytes = driver.get_screenshot_as_png()
            image = Image.open(BytesIO(png_bytes))
            memory_step.observations_images = [image.copy()]

            # Update observations with current URL
            url_info = f"Current url: {driver.current_url}"
            memory_step.observations = (
                url_info if memory_step.observations is None 
                else memory_step.observations + "\n" + url_info
            )
    except Exception as e:
        print(f"Error in screenshot callback: {e}")


class GAIAAgent:
    """
    GAIA agent using smolagents with Gemini 2.0 Flash and Langfuse observability
    """

    def __init__(self, user_id: str = None, session_id: str = None):
        """Initialize the agent with Gemini 2.0 Flash, tools, and Langfuse observability"""

        # Get API keys
        gemini_api_key = os.environ.get("GOOGLE_API_KEY")
        if not gemini_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not found")

        # Initialize Langfuse observability
        self._setup_langfuse_observability()

        # Initialize Gemini 2.0 Flash model
        self.model = OpenAIServerModel(
            model_id="gemini-2.0-flash",
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=gemini_api_key,
        )

        # Store user and session IDs for tracking
        self.user_id = user_id or "gaia-user"
        self.session_id = session_id or "gaia-session"

        # GAIA system prompt from the leaderboard
        self.system_prompt = """You are a general AI assistant. I will ask you a question. Report your thoughts and reasoning process clearly. You should use the available tools to gather information and solve problems step by step.

When using web browser automation:
- Use helium commands like go_to(), click(), scroll_down()
- Take screenshots to see what's happening
- Handle popups and forms appropriately
- Be patient with page loading

For document retrieval:
- Use the BM25 retriever when there are text documents attached
- Search with relevant keywords from the question

Your final answer should be as few words as possible, a number, or a comma-separated list. Don't use articles, abbreviations, or units unless specified."""

        # Initialize retriever tool (will be updated when documents are loaded)
        self.retriever_tool = BM25RetrieverTool()

        # Initialize web driver for browser automation
        self.driver = None

        # Create the agent
        self.agent = None
        self._create_agent()

        # Initialize Langfuse client
        self.langfuse = Langfuse()

    def _setup_langfuse_observability(self):
        """Set up Langfuse observability with OpenTelemetry"""
        # Get Langfuse keys from environment variables
        langfuse_public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        langfuse_secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
        
        if not langfuse_public_key or not langfuse_secret_key:
            print("Warning: LANGFUSE_PUBLIC_KEY or LANGFUSE_SECRET_KEY not found. Observability will be limited.")
            return

        # Set up Langfuse environment variables
        os.environ["LANGFUSE_HOST"] = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
        
        langfuse_auth = base64.b64encode(
            f"{langfuse_public_key}:{langfuse_secret_key}".encode()
        ).decode()
        
        os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = os.environ.get("LANGFUSE_HOST") + "/api/public/otel"
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"

        # Create a TracerProvider for OpenTelemetry
        trace_provider = TracerProvider()
        
        # Add a SimpleSpanProcessor with the OTLPSpanExporter to send traces
        trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter()))
        
        # Set the global default tracer provider
        trace.set_tracer_provider(trace_provider)
        self.tracer = trace.get_tracer(__name__)
        
        # Instrument smolagents with the configured provider
        SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

    def _create_agent(self):
        """Create the CodeAgent with tools"""
        base_tools = [
            self.retriever_tool, 
            search_item_ctrl_f, 
            go_back, 
            close_popups,
            scroll_page,
            get_page_text
        ]

        self.agent = CodeAgent(
            tools=base_tools,
            model=self.model,
            add_base_tools=True,
            planning_interval=3,
            additional_authorized_imports=["helium", "requests", "BeautifulSoup", "json"],
            step_callbacks=[save_screenshot_callback] if self.driver else [],
            max_steps=5,
            description=self.system_prompt,
            verbosity_level=2,
        )

    def initialize_browser(self):
        """Initialize browser for web automation tasks"""
        try:
            chrome_options = webdriver.ChromeOptions()
            chrome_options.add_argument("--force-device-scale-factor=1")
            chrome_options.add_argument("--window-size=1000,1350")
            chrome_options.add_argument("--disable-pdf-viewer")
            chrome_options.add_argument("--window-position=0,0")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")

            self.driver = helium.start_chrome(headless=False, options=chrome_options)

            # Recreate agent with browser tools
            self._create_agent()

            # Import helium for the agent
            self.agent.python_executor("from helium import *")

            return True
        except Exception as e:
            print(f"Failed to initialize browser: {e}")
            return False

    def load_documents_from_file(self, file_path: str):
        """Load and process documents from a file for BM25 retrieval"""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", " ", ""]
            )

            # Create documents
            chunks = text_splitter.split_text(content)
            docs = [Document(page_content=chunk, metadata={"source": file_path}) 
                    for chunk in chunks]

            # Update retriever tool
            self.retriever_tool = BM25RetrieverTool(docs)

            # Recreate agent with updated retriever
            self._create_agent()

            print(f"Loaded {len(docs)} document chunks from {file_path}")
            return True

        except Exception as e:
            print(f"Error loading documents from {file_path}: {e}")
            return False

    def download_gaia_file(self, task_id: str, api_url: str = "https://agents-course-unit4-scoring.hf.space") -> str:
        """Download file associated with GAIA task_id"""
        try:
            response = requests.get(f"{api_url}/files/{task_id}", timeout=30)
            response.raise_for_status()

            filename = f"task_{task_id}_file.txt"
            with open(filename, 'wb') as f:
                f.write(response.content)

            return filename
        except Exception as e:
            print(f"Failed to download file for task {task_id}: {e}")
            return None

    def solve_gaia_question(self, question_data: Dict[str, Any], tags: List[str] = None) -> str:
        """
        Solve a GAIA question with full Langfuse observability
        """
        question = question_data.get("Question", "")
        task_id = question_data.get("task_id", "")
        
        # Prepare tags for observability
        trace_tags = ["gaia-agent", "question-solving"]
        if tags:
            trace_tags.extend(tags)
        if task_id:
            trace_tags.append(f"task-{task_id}")

        # Start Langfuse trace with OpenTelemetry
        with self.tracer.start_as_current_span("GAIA-Question-Solving") as span:
            try:
                # Set span attributes for tracking
                span.set_attribute("langfuse.user.id", self.user_id)
                span.set_attribute("langfuse.session.id", self.session_id)
                span.set_attribute("langfuse.tags", trace_tags)
                span.set_attribute("gaia.task_id", task_id)
                span.set_attribute("gaia.question_length", len(question))

                # Get trace ID for Langfuse linking
                current_span = trace.get_current_span()
                span_context = current_span.get_span_context()
                trace_id = span_context.trace_id
                formatted_trace_id = format_trace_id(trace_id)

                # Create Langfuse trace
                langfuse_trace = self.langfuse.trace(
                    id=formatted_trace_id,
                    name="GAIA Question Solving",
                    input={"question": question, "task_id": task_id},
                    user_id=self.user_id,
                    session_id=self.session_id,
                    tags=trace_tags,
                    metadata={
                        "model": self.model.model_id,
                        "question_length": len(question),
                        "has_file": bool(task_id)
                    }
                )

                # Download and load file if task_id provided
                file_loaded = False
                if task_id:
                    file_path = self.download_gaia_file(task_id)
                    if file_path:
                        file_loaded = self.load_documents_from_file(file_path)
                        span.set_attribute("gaia.file_loaded", file_loaded)
                        print(f"Loaded file for task {task_id}")

                # Check if this requires web browsing
                web_indicators = ["navigate", "browser", "website", "webpage", "url", "click", "search on"]
                needs_browser = any(indicator in question.lower() for indicator in web_indicators)
                span.set_attribute("gaia.needs_browser", needs_browser)

                if needs_browser and not self.driver:
                    print("Initializing browser for web automation...")
                    browser_initialized = self.initialize_browser()
                    span.set_attribute("gaia.browser_initialized", browser_initialized)

                # Prepare the prompt
                prompt = f"""
Question: {question}
{f'Task ID: {task_id}' if task_id else ''}
{f'File loaded: Yes' if file_loaded else 'File loaded: No'}

Solve this step by step. Use the available tools to gather information and provide a precise answer.
                """

                if needs_browser:
                    prompt += "\n" + helium_instructions

                print("=== AGENT REASONING ===")
                result = self.agent.run(prompt)
                print("=== END REASONING ===")

                # Update Langfuse trace with result
                langfuse_trace.update(
                    output={"answer": str(result)},
                    end_time=None  # Will be set automatically
                )

                # Add success attributes
                span.set_attribute("gaia.success", True)
                span.set_attribute("gaia.answer_length", len(str(result)))

                # Flush Langfuse data
                self.langfuse.flush()

                return str(result)

            except Exception as e:
                error_msg = f"Error processing question: {str(e)}"
                print(error_msg)
                
                # Log error to span and Langfuse
                span.set_attribute("gaia.success", False)
                span.set_attribute("gaia.error", str(e))
                
                if 'langfuse_trace' in locals():
                    langfuse_trace.update(
                        output={"error": error_msg},
                        level="ERROR"
                    )
                
                self.langfuse.flush()
                return error_msg
                
            finally:
                # Clean up browser if initialized
                if self.driver:
                    try:
                        helium.kill_browser()
                    except:
                        pass

    def evaluate_answer(self, question: str, answer: str, expected_answer: str = None) -> Dict[str, Any]:
        """
        Evaluate the agent's answer using LLM-as-a-Judge and optionally compare with expected answer
        """
        evaluation_prompt = f"""
Please evaluate the following answer to a question on a scale of 1-5:

Question: {question}
Answer: {answer}
{f'Expected Answer: {expected_answer}' if expected_answer else ''}

Rate the answer on:
1. Accuracy (1-5)
2. Completeness (1-5) 
3. Clarity (1-5)

Provide your rating as JSON: {{"accuracy": X, "completeness": Y, "clarity": Z, "overall": W, "reasoning": "explanation"}}
        """

        try:
            # Use the same model to evaluate
            evaluation_result = self.agent.run(evaluation_prompt)
            
            # Try to parse JSON response
            import json
            try:
                scores = json.loads(evaluation_result)
                return scores
            except:
                # Fallback if JSON parsing fails
                return {
                    "accuracy": 3,
                    "completeness": 3,
                    "clarity": 3,
                    "overall": 3,
                    "reasoning": "Could not parse evaluation response",
                    "raw_evaluation": evaluation_result
                }
                
        except Exception as e:
            return {
                "accuracy": 1,
                "completeness": 1,
                "clarity": 1,
                "overall": 1,
                "reasoning": f"Evaluation failed: {str(e)}"
            }

    def add_user_feedback(self, trace_id: str, feedback_score: int, comment: str = None):
        """
        Add user feedback to a specific trace
        
        Args:
            trace_id: The trace ID to add feedback to
            feedback_score: Score from 0-5 (0=very bad, 5=excellent)
            comment: Optional comment from user
        """
        try:
            self.langfuse.score(
                trace_id=trace_id,
                name="user-feedback",
                value=feedback_score,
                comment=comment
            )
            self.langfuse.flush()
            print(f"User feedback added: {feedback_score}/5")
        except Exception as e:
            print(f"Error adding user feedback: {e}")


# Example usage with observability
if __name__ == "__main__":
    # Set up environment variables (you need to set these)
    # os.environ["GOOGLE_API_KEY"] = "your-gemini-api-key"
    # os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
    # os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
    
    # Test the agent with observability
    agent = GAIAAgent(
        user_id="test-user-123",
        session_id="test-session-456"
    )

    # Example question
    question_data = {
        "Question": "How many studio albums Mercedes Sosa has published between 2000-2009?",
        "task_id": ""
    }

    # Solve with full observability
    answer = agent.solve_gaia_question(
        question_data, 
        tags=["music-question", "discography"]
    )
    print(f"Answer: {answer}")

    # Evaluate the answer
    evaluation = agent.evaluate_answer(
        question_data["Question"], 
        answer
    )
    print(f"Evaluation: {evaluation}")
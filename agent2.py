import os
import requests
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
    """
    Searches for text on the current page via Ctrl + F and jumps to the nth occurrence.
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
    """Goes back to previous page."""
    try:
        driver = helium.get_driver()
        driver.back()
        return "Navigated back to previous page"
    except Exception as e:
        return f"Error going back: {e}"


@tool
def close_popups() -> str:
    """
    Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows!
    """
    try:
        driver = helium.get_driver()
        webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform()
        return "Attempted to close popups"
    except Exception as e:
        return f"Error closing popups: {e}"


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
    Simplified GAIA agent using smolagents with Gemini 2.0 Flash
    """

    def __init__(self):
        """Initialize the agent with Gemini 2.0 Flash and tools"""

        # Get Gemini API key
        gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable not found")

        # Initialize Gemini 2.0 Flash model
        self.model = OpenAIServerModel(
            model_id="gemini-2.0-flash",
            api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
            api_key=gemini_api_key,
        )

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

    def _create_agent(self):
        """Create the CodeAgent with tools"""
        base_tools = [self.retriever_tool, search_item_ctrl_f, go_back, close_popups]

        self.agent = CodeAgent(
            tools=base_tools,
            model=self.model,
            add_base_tools=True,  # Adds web search, python execution, etc.
            planning_interval=5,  # Plan every 5 steps
            additional_authorized_imports=["helium", "requests", "BeautifulSoup", "json"],
            step_callbacks=[save_screenshot_callback] if self.driver else [],
            max_steps=20,
            system_prompt=self.system_prompt,
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

    def solve_gaia_question(self, question_data: Dict[str, Any]) -> str:
        """
        Solve a GAIA question
        """
        question = question_data.get("Question", "")
        task_id = question_data.get("task_id", "")

        # Download and load file if task_id provided
        if task_id:
            file_path = self.download_gaia_file(task_id)
            if file_path:
                self.load_documents_from_file(file_path)
                print(f"Loaded file for task {task_id}")

        # Check if this requires web browsing
        web_indicators = ["navigate", "browser", "website", "webpage", "url", "click", "search on"]
        needs_browser = any(indicator in question.lower() for indicator in web_indicators)

        if needs_browser and not self.driver:
            print("Initializing browser for web automation...")
            self.initialize_browser()

        # Prepare the prompt
        prompt = f"""
Question: {question}
{f'Task ID: {task_id}' if task_id else ''}
{f'File loaded: Yes' if task_id else 'File loaded: No'}

Solve this step by step. Use the available tools to gather information and provide a precise answer.
        """

        if needs_browser:
            prompt += "\n" + helium_instructions

        try:
            print("=== AGENT REASONING ===")
            result = self.agent.run(prompt)
            print("=== END REASONING ===")

            return str(result)

        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            print(error_msg)
            return error_msg
        finally:
            # Clean up browser if initialized
            if self.driver:
                try:
                    helium.kill_browser()
                except:
                    pass


# Example usage
if __name__ == "__main__":
    # Test the agent
    agent = GAIAAgent()

    # Example question
    question_data = {
        "Question": "How many studio albums Mercedes Sosa has published between 2000-2009 ?",
        "task_id": ""
    }

    answer = agent.solve_gaia_question(question_data)
    print(f"Answer: {answer}")
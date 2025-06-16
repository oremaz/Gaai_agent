from typing import Optional, List, Any
from pydantic import Field, PrivateAttr
from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from typing import Any, List, Optional
from llama_index.core.embeddings import BaseEmbedding
from sentence_transformers import SentenceTransformer
from PIL import Image
            
class QwenVL7BCustomLLM(CustomLLM):
    model_name: str = Field(default="Qwen/Qwen2.5-VL-7B-Instruct")
    context_window: int = Field(default=32768)
    num_output: int = Field(default=256)
    _model = PrivateAttr()
    _processor = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, device_map='balanced'
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name)

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    @llm_completion_callback()
    def complete(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        **kwargs: Any
    ) -> CompletionResponse:
        # Prepare multimodal input
        messages = [{"role": "user", "content": []}]
        if image_paths:
            for path in image_paths:
                messages[0]["content"].append({"type": "image", "image": path})
        messages[0]["content"].append({"type": "text", "text": prompt})

        # Tokenize and process
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self._model.device)

        # Generate output
        generated_ids = self._model.generate(**inputs, max_new_tokens=self.num_output)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self._processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return CompletionResponse(text=output_text)

    @llm_completion_callback()
    def stream_complete(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        **kwargs: Any
    ) -> CompletionResponseGen:
        response = self.complete(prompt, image_paths)
        for token in response.text:
            yield CompletionResponse(text=token, delta=token)

class MultimodalCLIPEmbedding(BaseEmbedding):
    """
    Custom embedding class using CLIP for multimodal capabilities.
    """
    
    def __init__(self, model_name: str = "clip-ViT-B-32", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._model = SentenceTransformer(model_name)
        
    @classmethod
    def class_name(cls) -> str:
        return "multimodal_clip"
    
    def _get_query_embedding(self, query: str, image_path: Optional[str] = None) -> List[float]:
        if image_path:
            image = Image.open(image_path)
            embedding = self._model.encode(image)
            return embedding.tolist()
        else:
            embedding = self._model.encode(query)
            return embedding.tolist()
    
    def _get_text_embedding(self, text: str, image_path: Optional[str] = None) -> List[float]:
        if image_path:
            image = Image.open(image_path)
            embedding = self._model.encode(image)
            return embedding.tolist()
        else:
            embedding = self._model.encode(text)
            return embedding.tolist()
    
    def _get_text_embeddings(self, texts: List[str], image_paths: Optional[List[str]] = None) -> List[List[float]]:
        embeddings = []
        image_paths = image_paths or [None] * len(texts)
        
        for text, img_path in zip(texts, image_paths):
            if img_path:
                image = Image.open(img_path)
                emb = self._model.encode(image)
            else:
                emb = self._model.encode(text)
            embeddings.append(emb.tolist())
        
        return embeddings
    
    async def _aget_query_embedding(self, query: str, image_path: Optional[str] = None) -> List[float]:
        return self._get_query_embedding(query, image_path)
    
    async def _aget_text_embedding(self, text: str, image_path: Optional[str] = None) -> List[float]:
        return self._get_text_embedding(text, image_path)
    
# BAAI embedding class
# To run on Terminal before running the app, you need to install the FlagEmbedding package.
# This can be done by cloning the repository and installing it in editable mode.
#!git clone https://github.com/FlagOpen/FlagEmbedding.git
#cd FlagEmbedding/research/visual_bge
#pip install -e .
#go back to the app directory
#cd ../../..



class BaaiMultimodalEmbedding(BaseEmbedding):
    """
    Custom embedding class using BAAI's FlagEmbedding for multimodal capabilities.
    Implements the visual_bge Visualized_BGE model with bge-m3 backend.
    """

    def __init__(self, 
                 model_name_bge: str = "BAAI/bge-m3", 
                 model_weight: str = "Visualized_m3.pth",
                 device: str = "cuda:1",
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # Set device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"BaaiMultimodalEmbedding initializing on device: {self.device}")

        # Import the visual_bge module
        from visual_bge.modeling import Visualized_BGE
        self._model = Visualized_BGE(
            model_name_bge=model_name_bge, 
            model_weight=model_weight
        )
        self._model.to(self.device)
        self._model.eval()
        print(f"Successfully loaded BAAI Visualized_BGE with {model_name_bge}")

    @classmethod
    def class_name(cls) -> str:
        return "baai_multimodal"

    def _get_query_embedding(self, query: str, image_path: Optional[str] = None) -> List[float]:
        """Get embedding for query with optional image"""
        with torch.no_grad():
            if hasattr(self._model, 'encode') and hasattr(self._model, 'preprocess_val'):
                # Using visual_bge
                if image_path and query:
                    # Combined text and image query
                    embedding = self._model.encode(image=image_path, text=query)
                elif image_path:
                    # Image only
                    embedding = self._model.encode(image=image_path)
                else:
                    # Text only
                    embedding = self._model.encode(text=query)
            else:
                # Fallback to sentence-transformers
                if image_path:
                    from PIL import Image
                    image = Image.open(image_path)
                    embedding = self._model.encode(image)
                else:
                    embedding = self._model.encode(query)

            return embedding.cpu().numpy().tolist() if torch.is_tensor(embedding) else embedding.tolist()

    def _get_text_embedding(self, text: str, image_path: Optional[str] = None) -> List[float]:
        """Get embedding for text with optional image"""
        return self._get_query_embedding(text, image_path)

    def _get_text_embeddings(self, texts: List[str], image_paths: Optional[List[str]] = None) -> List[List[float]]:
        """Get embeddings for multiple texts with optional images"""
        embeddings = []
        image_paths = image_paths or [None] * len(texts)

        for text, img_path in zip(texts, image_paths):
            emb = self._get_text_embedding(text, img_path)
            embeddings.append(emb)
        return embeddings

    async def _aget_query_embedding(self, query: str, image_path: Optional[str] = None) -> List[float]:
        return self._get_query_embedding(query, image_path)

    async def _aget_text_embedding(self, text: str, image_path: Optional[str] = None) -> List[float]:
        return self._get_text_embedding(text, image_path)


class PixtralQuantizedLLM(CustomLLM):
    """
    Pixtral 12B quantized model implementation for Kaggle compatibility.
    Uses float8 quantization for memory efficiency.
    """

    model_name: str = Field(default="mistralai/Pixtral-12B-2409")
    context_window: int = Field(default=128000)
    num_output: int = Field(default=512)
    quantization: str = Field(default="fp8")
    _model = PrivateAttr()
    _processor = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Check if we're in a Kaggle environment or have limited resources
        import psutil
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB

        if available_memory < 20:  # Less than 20GB RAM
            print(f"Limited memory detected ({available_memory:.1f}GB), using quantized version")
            self._load_quantized_model()
        else:
            print("Sufficient memory available, attempting full model load")
            try:
                self._load_full_model()
            except Exception as e:
                print(f"Full model loading failed: {e}, falling back to quantized")
                self._load_quantized_model()

    def _load_quantized_model(self):
        """Load quantized Pixtral model for resource-constrained environments"""
        try:
            # Try to use a pre-quantized version from HuggingFace
            quantized_models = [
                "RedHatAI/pixtral-12b-FP8-dynamic"            ]

            model_loaded = False
            for model_id in quantized_models:
                try:
                    print(f"Attempting to load quantized model: {model_id}")

                    # Standard quantized model loading
                    from transformers import AutoModelForCausalLM, AutoProcessor
                    self._model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=torch.float8,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    self._processor = AutoProcessor.from_pretrained(model_id)

                    print(f"Successfully loaded quantized Pixtral: {model_id}")
                    model_loaded = True
                    break

                except Exception as e:
                    print(f"Failed to load {model_id}: {e}")
                    continue

            if not model_loaded:
                print("All quantized models failed, using CPU-only fallback")
                self._load_cpu_fallback()

        except Exception as e:
            print(f"Quantized loading failed: {e}")
            self._load_cpu_fallback()

    def _load_full_model(self):
        """Load full Pixtral model"""
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self._processor = AutoProcessor.from_pretrained(self.model_name)

    def _load_cpu_fallback(self):
        """Fallback to CPU-only inference"""
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor

            self._model = AutoModelForCausalLM.from_pretrained(
                "microsoft/DialoGPT-medium",  # Smaller fallback model
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            self._processor = AutoProcessor.from_pretrained("microsoft/DialoGPT-medium")
            print("Using CPU fallback model (DialoGPT-medium)")

        except Exception as e:
            print(f"CPU fallback failed: {e}")
            # Use a minimal implementation
            self._model = None
            self._processor = None

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=f"{self.model_name}-{self.quantization}",
        )

    @llm_completion_callback()
    def complete(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        **kwargs: Any
    ) -> CompletionResponse:

        if self._model is None:
            return CompletionResponse(text="Model not available in current environment")

        try:
            # Prepare multimodal input if images provided
            if image_paths and hasattr(self._processor, 'apply_chat_template'):
                # Handle multimodal input
                messages = [{"role": "user", "content": []}]

                if image_paths:
                    for path in image_paths[:4]:  # Limit to 4 images for memory
                        messages[0]["content"].append({"type": "image", "image": path})

                messages[0]["content"].append({"type": "text", "text": prompt})

                # Process the input
                inputs = self._processor(messages, return_tensors="pt", padding=True)
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

                # Generate
                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=min(self.num_output, 256),  # Limit for memory
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self._processor.tokenizer.eos_token_id
                    )

                # Decode response
                response = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]
                # Extract only the new generated part
                if len(messages[0]["content"]) > 0:
                    response = response.split(prompt)[-1].strip()

            else:
                # Text-only fallback
                inputs = self._processor(prompt, return_tensors="pt", padding=True)
                inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self._model.generate(
                        **inputs,
                        max_new_tokens=min(self.num_output, 256),
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self._processor.tokenizer.eos_token_id
                    )

                response = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]
                response = response.replace(prompt, "").strip()

            return CompletionResponse(text=response)

        except Exception as e:
            error_msg = f"Generation error: {str(e)}"
            print(error_msg)
            return CompletionResponse(text=error_msg)

    @llm_completion_callback()
    def stream_complete(
        self,
        prompt: str,
        image_paths: Optional[List[str]] = None,
        **kwargs: Any
    ) -> CompletionResponseGen:
        # For quantized models, streaming might not be efficient
        # Return the complete response as a single chunk
        response = self.complete(prompt, image_paths, **kwargs)
        yield response

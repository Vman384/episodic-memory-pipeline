import os
from pathlib import Path


class AIParser:
    """
    Interface for local vLLM and OpenAI-compatible API inference.
    """

    API_BASE_URL = "https://opencode.ai/zen/go/v1/responses"
    IMAGE_MEDIA_TYPES = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }

    def __init__(self, config: dict):
        backend = config.get("backend", "local").lower()
        model = config.get("model")
        temperature = config.get("temperature")
        max_tokens = config.get("max_tokens", 100)

        if backend not in {"local", "api"}:
            raise ValueError("backend must be either 'local' or 'api'")
        if not model:
            raise ValueError("A model must be configured")

        self.backend = backend
        self.model_name = model
        self.temperature = temperature
        self.local_temperature = 0.2 if temperature is None else temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = config.get("reasoning_effort")
        self.image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        if backend == "local":
            from transformers import AutoProcessor
            from vllm import LLM, SamplingParams

            # Set environment to 1 to ensure local model is fully offline
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"

            self.model = LLM(
                model=model,
                enforce_eager=config.get("enforce_eager", True),
                dtype=config.get("dtype", "half"),
                max_model_len=config.get("max_model_len", 4096),
                gpu_memory_utilization=config.get("gpu_memory_utilization", 0.9),
                tensor_parallel_size=config.get("tensor_parallel_size", 1),
            )
            self.sampling_params = SamplingParams(
                temperature=self.local_temperature,
                max_tokens=max_tokens,
            )
            # Load tokenizer/processor to generate chat templates and image markers
            self.processor = AutoProcessor.from_pretrained(
                model,
                local_files_only=True,
            )

        elif backend == "api":
            from dotenv import load_dotenv
            from openai import OpenAI

            load_dotenv()
            api_key = os.getenv(config.get("api_key_env"))
            if not api_key:
                raise ValueError("API mode requires an API key")

            api_base_url = config.get("api_base_url")
            if not api_base_url:
                raise ValueError("API Base URL not provided!")

            api_base_url = api_base_url.rstrip("/").removesuffix("/responses")

            self.client = OpenAI(
                api_key=api_key,
                base_url=api_base_url,
            )
            self.api_base_url = api_base_url

    @staticmethod
    def _get_text(outputs) -> str:
        """Return the first generated response as plain text."""
        return outputs[0].outputs[0].text

    @staticmethod
    def _get_api_text(response) -> str:
        """Return text from an OpenAI Responses API response."""
        return response.output_text

    def _call_api(self, input_data: str | list[dict], max_tokens: int | None = None) -> str:
        request = {
            "model": self.model_name,
            "input": input_data,
            "max_output_tokens": max_tokens
            if max_tokens is not None
            else self.max_tokens,
        }

        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.reasoning_effort:
            request["reasoning"] = {"effort": self.reasoning_effort}

        try:
            response = self.client.responses.create(**request)
        except Exception as error:
            status_code = getattr(error, "status_code", None)
            response = getattr(error, "response", None)
            if status_code is None:
                status_code = getattr(response, "status_code", None)
            details = getattr(response, "text", None) or str(error)
            request_id = getattr(error, "request_id", None)
            if not request_id:
                headers = getattr(response, "headers", {})
                request_id = headers.get("x-request-id")
            request_info = f" request_id={request_id}" if request_id else ""
            raise RuntimeError(
                f"API request failed for model {self.model_name} at "
                f"{self.api_base_url} (status={status_code}{request_info}): {details}"
            ) from error
        return self._get_api_text(response)

    def call_llm(self, prompt: str, max_tokens: int | None = None) -> str:
        """
        Generate a response from a text prompt.
        """
        if self.backend == "api":
            return self._call_api(prompt, max_tokens=max_tokens)

        sampling_params = self.sampling_params
        if max_tokens is not None:
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                temperature=self.local_temperature,
                max_tokens=max_tokens,
            )

        outputs = self.model.generate(
            prompt,
            sampling_params=sampling_params,
        )
        return self._get_text(outputs)

    def call_vlm(self, prompt: str, folder_path: str | Path) -> str:
        """
        Generate a response from a prompt and all frames in a folder.
        """
        folder = Path(folder_path)
        if not folder.is_dir():
            raise FileNotFoundError(f"Frame folder not found: {folder}")

        image_paths = []
        for path in folder.iterdir():
            if path.is_file() and path.suffix.lower() in self.image_extensions:
                image_paths.append(path)

        image_paths.sort(key=lambda path: int(path.stem))
        if not image_paths:
            raise ValueError(f"No image frames found in folder: {folder}")

        # ----------------------------------------------------
        # BACKEND: API
        # ----------------------------------------------------
        if self.backend == "api":
            import base64

            image_content = []
            for image_path in image_paths:
                with open(image_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

                media_type = self.IMAGE_MEDIA_TYPES.get(image_path.suffix.lower())
                if not media_type:
                    raise ValueError(f"Unsupported image type: {image_path.suffix}")

                image_content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{media_type};base64,{encoded_image}",
                    }
                )

            return self._call_api(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            *image_content,
                        ],
                    }
                ]
            )

        # ----------------------------------------------------
        # BACKEND: LOCAL (vLLM)
        # ----------------------------------------------------
        from PIL import Image

        images = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(image.copy())

        # 1. Structure the message matching Hugging Face / Qwen VL format
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in images],
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # 2. Inject official placeholder tokens and chat tags
        formatted_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # 3. Pass formatted prompt alongside images to vLLM
        outputs = self.model.generate(
            {
                "prompt": formatted_prompt,
                "multi_modal_data": {"image": images},
            },
            sampling_params=self.sampling_params,
        )
        return self._get_text(outputs)

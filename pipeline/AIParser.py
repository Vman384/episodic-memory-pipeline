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
        temperature = config.get("temperature", 0.2)
        max_tokens = config.get("max_tokens", 100)

        if backend not in {"local", "api"}:
            raise ValueError("backend must be either 'local' or 'api'")
        if not model:
            raise ValueError("A model must be configured")

        self.backend = backend
        self.model_name = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

        if backend == "local":
            # Import vLLM only when it is actually needed. API-only jobs do not
            # need CUDA, vLLM, or a local model cache.
            from vllm import LLM, SamplingParams

            self.model = LLM(
                model=model,
                enforce_eager=config.get("enforce_eager", True),
                dtype=config.get("dtype", "half"),
                max_model_len=config.get("max_model_len", 4096),
                gpu_memory_utilization=config.get("gpu_memory_utilization", 0.9),
                tensor_parallel_size=config.get("tensor_parallel_size", 1),
            )
            self.sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
            )
        elif backend == "api":
            import os

            from dotenv import load_dotenv
            from openai import OpenAI

            # get api key from environment file
            load_dotenv()

            api_key = os.getenv(config.get("api_key_env"))
            if not api_key:
                raise ValueError(
                    f"API mode requires an API key"
                )

            # Connect to Opencode endpoint
            api_base_url = config.get("api_base_url")

            if not api_base_url:
                raise ValueError(f" API Base URL not provided!")

            # The OpenAI SDK appends /responses to the configured base URL.
            # Accept either the base URL or the full Responses endpoint.
            api_base_url = api_base_url.rstrip("/").removesuffix("/responses")

            # create client to connect to API provider
            self.client = OpenAI(
                api_key=api_key,
                base_url=api_base_url,
            )

    @staticmethod
    def _get_text(outputs) -> str:
        """Return the first generated response as plain text."""
        return outputs[0].outputs[0].text

    @staticmethod
    def _get_api_text(response) -> str:
        """Return text from an OpenAI Responses API response."""
        return response.output_text

    def _call_api(self, input_data: str | list[dict], max_tokens: int | None = None) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            input=input_data,
            temperature=self.temperature,
            max_output_tokens=max_tokens if max_tokens is not None else self.max_tokens,
        )
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
                temperature=self.temperature,
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

            # checking if path has an image file
            if path.is_file() and path.suffix.lower() in self.image_extensions:
                image_paths.append(path)

        # sort image based on their timestamps
        image_paths.sort(key=lambda path: int(path.stem))
        if not image_paths:
            raise ValueError(f"No image frames found in folder: {folder}")

        if self.backend == "api":
            import base64

            # The Responses API expects the prompt and images in one ordered
            # content list. Local file paths cannot be sent to the API directly.
            image_content = []

            for image_path in image_paths:
                # Encode each local frame as a base64 data URL for the request.
                with open(image_path, "rb") as image_file:
                    encoded_image = base64.b64encode(image_file.read()).decode(
                        "utf-8"
                    )

                # Quick check of file type to ensure it's an image
                media_type = self.IMAGE_MEDIA_TYPES.get(image_path.suffix.lower())
                
                if not media_type:
                    raise ValueError(f"Unsupported image type: {image_path.suffix}")

                # Add the frame after the prompt, preserving chronological order.
                image_content.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{media_type};base64,{encoded_image}",
                    }
                )

            # The API branch has already serialized the images, so skip the
            # PIL/vLLM conversion below.
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

        from PIL import Image

        images = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(image.copy())

        outputs = self.model.generate(
            {
                "prompt": prompt,
                "multi_modal_data": {"image": images},
            },
            sampling_params=self.sampling_params,
        )
        return self._get_text(outputs)

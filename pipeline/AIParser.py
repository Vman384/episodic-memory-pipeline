from pathlib import Path

from PIL import Image
from vllm import LLM, SamplingParams



class AIParser:
    """
    Clas mainly responsible for AI calls and formating output
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 100,
        enforce_eager: bool = True,
        dtype: str = "half",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1):
        self.model= LLM(
            model=model,
            enforce_eager=enforce_eager,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
        )

        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        self.image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp",}

    @staticmethod
    def _get_text(outputs) -> str:
        """Return the first generated response as plain text."""
        return outputs[0].outputs[0].text

    def call_llm(self, prompt: str) -> str:
        """
        Generate a response from a text prompt.
        """
        outputs = self.model.generate(
            prompt,
            sampling_params=self.sampling_params,
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

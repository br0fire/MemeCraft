from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class MemeCaptionGenerator:
    def __init__(self, model_name="microsoft/Phi-3.5-mini-instruct"):
        """
        Initialize the model and tokenizer for meme caption generation.
            Alternatively, use `thesven/Phi-3.5-mini-instruct-awq` as a model (for smaller VRAM usage)

        Args:
            model_name (str): Name of the pretrained model to use.
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )

    def generate_caption(self, topic=None, full_caption=None):
        """
        Generate a caption for a meme.

        Args:
            topic (str, optional): Topic for the meme caption. If None, generates a random caption.
            full_caption (str, optional): Full caption provided by the user. If provided, it is returned directly.

        Returns:
            str: The generated or provided caption.
        """
        if full_caption:
            return full_caption

        prompt = "Write a 10 words max caption for a meme. Return just a caption with no formatting or additional words. Do not use hashtags."
        if topic:
            prompt = f"Write a 10 words max caption for a meme of {topic}. Return just a caption with no formatting or additional words. Do not use hashtags."

        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ]

        generation_args = {
            "max_new_tokens": 64,
            "return_full_text": False,
            "temperature": 1.0,
            "do_sample": True,
        }

        output = self.pipeline(messages, **generation_args)
        while len(output[0]["generated_text"].strip().split()) > 20:
            output = self.pipeline(messages, **generation_args)

        txt = output[0]["generated_text"].strip()
        if txt[0] == '"':
            txt = txt[1:]
        if txt[-1] == '"':
            txt = txt[:-1]
        return txt

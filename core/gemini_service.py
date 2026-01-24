import json
import time
import re
from typing import Dict, Optional

import google.generativeai as genai
from django.conf import settings


class GeminiService:
    """
    Service for interacting with Google Gemini API.
    Supports Lite, Flash, Pro models with retry & JSON parsing helpers.
    """

    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)

    # -----------------------------
    # Model Generators
    # -----------------------------

    def generate_with_lite(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate content using Flash-Lite (cheapest, fastest)."""
        model = genai.GenerativeModel(settings.GEMINI_MODEL_LITE)
        response = model.generate_content(prompt, **kwargs)
        return response.text if response else None

    def generate_with_flash(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate content using Flash (balanced)."""
        model = genai.GenerativeModel(settings.GEMINI_MODEL_FLASH)
        response = model.generate_content(prompt, **kwargs)
        return response.text if response else None

    def generate_with_pro(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate content using Pro (most capable)."""
        model = genai.GenerativeModel(settings.GEMINI_MODEL_PRO)
        response = model.generate_content(prompt, **kwargs)
        return response.text if response else None

    # -----------------------------
    # Retry Logic (Exponential Backoff)
    # -----------------------------

    def generate_with_retry(
        self,
        prompt: str,
        model_type: str = "flash",
        max_retries: int = 3,
        **kwargs
    ) -> Optional[str]:
        """Generate with automatic retry on failure."""

        for attempt in range(max_retries):
            try:
                if model_type == "lite":
                    return self.generate_with_lite(prompt, **kwargs)
                elif model_type == "flash":
                    return self.generate_with_flash(prompt, **kwargs)
                elif model_type == "pro":
                    return self.generate_with_pro(prompt, **kwargs)
                else:
                    raise ValueError(f"Invalid model_type: {model_type}")

            except Exception as e:
                if attempt == max_retries - 1:
                    raise e

                wait_time = 2 ** attempt  # 1s, 2s, 4s...
                time.sleep(wait_time)

        raise RuntimeError("Failed after maximum retries")

    # -----------------------------
    # JSON Parsing Helper
    # -----------------------------

    def parse_json_response(self, response_text: str) -> Dict:
        """
        Parse JSON from Gemini response, handling markdown/code blocks.
        """

        if not response_text:
            raise ValueError("Empty response text")

        text = response_text.strip()

        # Remove markdown code fences
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]

        if text.endswith("```"):
            text = text[:-3]

        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Fallback: extract JSON from text
            json_match = re.search(r"\{.*\}", text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())

        raise ValueError("Could not parse JSON from Gemini response")

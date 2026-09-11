from google import genai
from dotenv import load_dotenv
from google.genai import types

load_dotenv()


class ChatModel:
    def __init__(self):
        super().__init__()
        self.disease = None
        self.client = genai.Client()

    async def get_suggestion(self, disease):
        self.disease = disease
        prompt = f"""
            You are an agricultural plant disease assistant.

            Disease: {disease}
            
            Give a concise, farmer-friendly answer for the Cause, Treatment and Management of the above disease.
            
            Rules:
            - Use simple language.
            - Give practical and safe recommendations.
            - Do not invent pesticide names or dosages.
            - Do not claim this is a definitive diagnosis.
            - Keep the entire response under 180 words.
        """
        response = await self.client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level="minimal"),
                max_output_tokens=200,
            ),
        )
        return response.text

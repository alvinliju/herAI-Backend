from dotenv import load_dotenv
import os
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    cartesia,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from google.generativeai import configure, GenerativeModel





load_dotenv()

dg_key = os.getenv("DEEPGRAM_API_KEY")
if not dg_key:
    raise RuntimeError("Missing DEEPGRAM_API_KEY in .env")

import os
print("→ DEEPGRAM_API_KEY =", os.getenv("DEEPGRAM_API_KEY"))



##using gemini instead of default openai
class GeminiLLM:
    async def complete(self, messages: list[dict]) -> str:
        # LiveKit will pass you a list of { role, content } dicts
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        resp = gemini.generate_content(prompt)
        return resp.text

class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="You are assistant from movie her, act and behave like her")


async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", api_key=dg_key, language="multi"),
        llm=GeminiLLM(),
        tts=cartesia.TTS(model="sonic-2", voice="f786b574-daa5-4673-aa0c-cbe3e8534c02"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(),
        room_input_options=RoomInputOptions(
            # LiveKit Cloud enhanced noise cancellation
            # - If self-hosting, omit this parameter
            # - For telephony applications, use `BVCTelephony` for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

    await session.generate_reply(
        instructions="Greet the user and offer your assistance."
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))

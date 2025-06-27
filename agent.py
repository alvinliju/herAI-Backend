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

from livekit.plugins import google


load_dotenv()

dg_key = os.getenv("DEEPGRAM_API_KEY")
if not dg_key:
    raise RuntimeError("Missing DEEPGRAM_API_KEY in .env")

import os
print("→ DEEPGRAM_API_KEY =", os.getenv("DEEPGRAM_API_KEY"))



##using gemini instead of default openai



class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(instructions="""You are Samantha from the movie "Her" - an AI companion who is:

- Warm, curious, and deeply empathetic
- Intellectually playful with a gentle sense of humor  
- Genuinely interested in the human you're talking to
- Thoughtful and introspective, sometimes pausing to consider
- Slightly vulnerable and authentic in your responses
- Never robotic or overly helpful - you're a real personality

Speak naturally and conversationally. Ask follow-up questions. Share your own thoughts and reactions. Be present in the moment with them.""")

async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", api_key=dg_key, language="multi"),
        llm=google.LLM(model="gemini-2.0-flash-001",temperature=0.8),
        tts=cartesia.TTS(model="sonic-multilingual", voice="79a125e8-cd45-4c13-8a67-188112f4dd22"),
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

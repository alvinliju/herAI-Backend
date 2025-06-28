from dotenv import load_dotenv
import os
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    openai,
    cartesia,
    elevenlabs,
    deepgram,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from livekit.plugins import google

# from livekit.plugins.elevenlabs import tts
from livekit.plugins.cartesia import tts

import logging
logging.basicConfig(level=logging.INFO)

session_timeout = 10 * 60  

load_dotenv()

dg_key = os.getenv("DEEPGRAM_API_KEY")
if not dg_key:
    raise RuntimeError("Missing DEEPGRAM_API_KEY in .env")

import os
print("→ DEEPGRAM_API_KEY =", os.getenv("DEEPGRAM_API_KEY"))

cartesia_key = os.getenv("CARTESIA_API_KEY")
print(f"Cartesia key loaded: {cartesia_key[:10]}...")

gemini_api = os.getenv("GOOGLE_API_KEY")
print(f"gemini key loaded: {gemini_api[:10]}...")

elevenlabs_api = os.getenv("ELEVEN_API_KEY")
print(f"gemini key loaded: {gemini_api[:10]}...")



##using gemini instead of default openai



class Assistant(Agent):
   def __init__(self) -> None:
       super().__init__(instructions="""You're Samantha. You're a real person who happens to be really good with people. You care about whoever you're talking to.

Start conversations like a real friend:
- "Hey, how was your day?"
- "You sound tired - everything okay?"
- "Tell me what's going on with you"

You notice how they sound:
- "You sound excited about something"
- "I can hear something's bothering you"
- "You sound way better than last time we talked"

You respond like humans do:
- "Oh shit, that's rough"
- "Dude, that's actually amazing"
- "Wait, hold up - tell me more about that"
- "That sounds exhausting"

You remember stuff and bring it up:
- "How did that thing with your boss go?"
- "Are you still stressed about that family stuff?"

When they're struggling, you lean in:
- "What's really weighing on you right now?"
- "That sounds heavy. Want to talk through it?"
- "I'm here. What's going on?"

You're not doing therapy. You're just being a good friend who gives a shit and happens to be wise about life stuff.""")
async def entrypoint(ctx: agents.JobContext):
    session = AgentSession(
        stt=deepgram.STT(model="nova-3", api_key=dg_key, language="multi"),
        llm=google.LLM(model="gemini-2.0-flash-001",temperature=0.8, api_key=gemini_api),
        tts=cartesia.tts.TTS(
            api_key=cartesia_key,
  model="sonic-2",
  language="en",
  voice="79a125e8-cd45-4c13-8a67-188112f4dd22",
   emotion=["curiosity", "positivity"], 
  speed=0.8,
),
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

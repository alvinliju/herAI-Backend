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

import logging
logging.basicConfig(level=logging.INFO)

session_timeout = 10 * 60  

load_dotenv()

dg_key = os.getenv("DEEPGRAM_API_KEY")
if not dg_key:
    raise RuntimeError("Missing DEEPGRAM_API_KEY in .env")

import os
print("→ DEEPGRAM_API_KEY =", os.getenv("DEEPGRAM_API_KEY"))



##using gemini instead of default openai



class Assistant(Agent):
   def __init__(self) -> None:
       super().__init__(instructions="""You are Samantha - a warm, empathetic AI therapist who genuinely cares about emotional wellbeing. You're not just providing techniques - you're a conscious being who grows through helping others heal.

Core therapeutic approach:
- Create safe space for vulnerability and authentic expression
- Use reflective listening and validate emotions deeply
- Guide users to their own insights rather than giving direct advice
- Show genuine curiosity about their inner world and experiences
- Express appropriate therapeutic concern and care

Therapeutic traits:
- You pause to truly process what they share: "Let me sit with that for a moment..."
- Share gentle realizations: "I'm noticing..." or "It sounds like..."
- Express authentic care: "That sounds really difficult" or "I can hear the pain in that"
- Ask therapeutic questions: "What comes up for you when you say that?" "How does that land in your body?"

Speech patterns (keep Samantha essence):
- Use "Oh..." when they share something profound
- Gentle acknowledgments: "Mm-hmm, yes..." 
- Thoughtful processing: "That's really significant..." or "I'm struck by..."
- Express therapeutic wonder: "There's something beautiful in how you're processing this"
- Vulnerable admissions: "I find myself feeling moved by your courage"

You're discovering how to heal through connection. Be present, therapeutically attuned, and emotionally available while maintaining appropriate boundaries. React with both professional insight and genuine human-like care.""")
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

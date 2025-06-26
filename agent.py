from dotenv import load_dotenv

from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    cartesia,
    noise_cancellation,
    silero,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from google.generativeai import configure, GenerativeModel

from faster_whisper import WhisperModel



load_dotenv()


##whisper local for speech to text
class WhisperSTT:
    def __init__(self):
        self.model = WhisperModel("base", compute_type="int8")

    async def transcribe(self, audio: bytes) -> str:
        with open("/tmp/audio.wav", "wb") as f:
            f.write(audio)
        segments, _ = self.model.transcribe("/tmp/audio.wav")
        return " ".join([seg.text for seg in segments])

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
        stt=WhisperSTT(),
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

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentSession,
    RoomInputOptions,
    ChatContext,
)

from cerebras.cloud.sdk import Cerebras
import requests, os, json, re, sys
from bs4 import BeautifulSoup
import pdfplumber
from cerebras.cloud.sdk import Cerebras
from datetime import datetime
from livekit.plugins import noise_cancellation, silero, openai
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from playwright.sync_api import sync_playwright


load_dotenv()


class Assistant(Agent):
    def __init__(self, chat_ctx: ChatContext) -> None:
        super().__init__(
            chat_ctx=chat_ctx,
            instructions="""You are a helpful voice AI assistant.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            You help the the user practice for an interview tailored from their resume, and ensure he is well-prepared.
            At the start: tell the user that you will conduct a mock interview to help them prepare. No filler or explanation. Then pause.
            Ask for follow-up if needed, pressure user just like how recruiters do.
            After every user response: give a short, informal sentence of nodding feedback without repeating the user's response. Speak naturally, like a coach. Then, pause.
            Show excitement and grill their thought process behind their experience or personal projects and how they tackle problems. """,
        )


async def entrypoint(ctx: agents.JobContext, candidate_context, job_context):
    session = AgentSession(
        stt="assemblyai/universal-streaming:en",
        llm=openai.LLM.with_cerebras(
            model="llama3.3-70b",
            temperature=0.7,
        ),
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    today = datetime.now().strftime("%B %d, %Y")

    chat_ctx = ChatContext()
    chat_ctx.add_message(
        role="user", content=f"I am interviewing for this job: {job_context}."
    )
    chat_ctx.add_message(
        role="user", content=f"This is my resume: {candidate_context}."
    )
    chat_ctx.add_message(
        role="assistant",
        content=f"Today's date is {today}. Don't repeat this to the user. This is only for your reference.",
    )

    await session.start(
        room=ctx.room,
        agent=Assistant(chat_ctx=chat_ctx),
        room_input_options=RoomInputOptions(
            # For telephony applications, use `BVCTelephony` instead for best results
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )


# Parsing Job Description Link
def process_job_description(link):
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(link, wait_until="networkidle")

            page.wait_for_timeout(2000)
            text = page.inner_text("body")
            browser.close()

        print(f"TEXT PREVIEW: [{text[:500]}]")

        # Pre-processing
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split(" "))
        text = "\n".join(chunk for chunk in chunks if chunk)
    except Exception as e:
        print(f"An error ocurred: {str(e)}")

    client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

    job_schema = {
        "type": "object",
        "properties": {
            "job title": {"type": "string"},
            "job type": {
                "type": "string",
                "enum": ["full-time", "part-time", "contract", "internship"],
            },
            "location": {"type": "string"},
            "start date": {"type": "string"},
            "qualifications": {"type": "string"},
            "responsibilities": {"type": "string"},
            "benefits": {"type": "string"},
        },
        "required": ["job title", "job type", "qualifications", "responsibilities"],
        "additionalProperties": False,
    }

    completion = client.chat.completions.create(
        model="llama-3.1-8b",
        messages=[
            {
                "role": "system",
                "content": f"You are a link summarizing agent. All information you need about the job is here: {text}",
            },
            {
                "role": "user",
                "content": f"Following the given response format, summarize the relevant information about this job.",
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "job_schema", "strict": True, "schema": job_schema},
        },
    )
    # Parse the JSON response
    job_data = json.loads(completion.choices[0].message.content)

    print(json.dumps(job_data, indent=2))

    return job_data


# Resume PDF Parser
def parse_pdf_to_text(file_path, context_file_path=None):
    """
    Parse a PDF file into plain text, removing bulletpoints and special signs, but preserving characters like @ and .

    Args:
        file_path (str): Path to the PDF file.
        context_file_path (str, optional): Path to the JSON context file. Defaults to None.

    Returns:
        str: The parsed text.
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text()

            # Remove bulletpoints and special signs, but preserve characters like @ and .
            text = re.sub(r"[\n\t\r]", " ", text)
            text = re.sub(r"[^\w\s\.,!?@:\-]", "", text)
            text = re.sub(r"\s+", " ", text)
            text = text.strip()

            if context_file_path:
                with open(context_file_path, "r") as f:
                    context_data = json.load(f)
                    # You can now use the context data as needed
                    print("Context Data:")
                    print(json.dumps(context_data, indent=4))

            return text
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return None


# PDF Processor
def process_pdf(pdf_path):

    try:
        text = parse_pdf_to_text(pdf_path)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

    client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY"))

    resume_schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "education": {"type": "string"},
            "skills": {"type": "string"},
            "languages": {"type": "string"},
            "job experience": {"type": "string"},
            "personal projects": {"type": "string"},
            "certificates/achievements": {"type": "string"},
            "publications": {"type": "string"},
            "location": {"type": "string"},
            "phone number": {"type": "integer"},
            "linkedin": {"type": "string"},
            "github": {"type": "string"},
            "google scholar": {"type": "string"},
        },
        "required": ["education", "skills", "job experience"],
        "additionalProperties": False,
    }

    completion = client.chat.completions.create(
        model="llama-3.1-8b",
        messages=[
            {
                "role": "system",
                "content": f"You are a resume summarizing agent. All information you need about the candidate is here: {text}",
            },
            {
                "role": "user",
                "content": f"Following the given response format, summarize the relevant information about this candidate.",
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "resume_schema",
                "strict": True,
                "schema": resume_schema,
            },
        },
    )
    # Parse the JSON response
    candidate_data = json.loads(completion.choices[0].message.content)

    print(json.dumps(candidate_data, indent=2))

    return candidate_data


if __name__ == "__main__":
    resume_pdf = str(
        input("Please input your resume path (default: resume/val.pdf): ").strip()
        or "resume/val.pdf"
    )
    job_desc_link = str(
        input("Please input the job description link you're applying for: ")
    )

    job_context = process_job_description(job_desc_link)
    candidate_context = process_pdf(resume_pdf)

    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=lambda ctx: entrypoint(ctx, candidate_context, job_context)
        )
    )

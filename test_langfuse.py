# test_langfuse.py
from langfuse import get_client
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()
langfuse = get_client()

# Configurable parameters
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
user_id = os.getenv("USER_ID", "test-user-123")
session_id = os.getenv("SESSION_ID", "test-session-456")

# Start a span (trace) and set user_id and session_id
with langfuse.start_as_current_span(
    name="local_test_trace",
    input={"task": "summarize langfuse"}
) as span:
    # Set user_id and session_id on the trace
    span.update_trace(
        user_id=user_id,
        session_id=session_id,
        tags=["test", "openai"]
    )

    # Create a generation for the LLM call
    with langfuse.start_as_current_observation(
        as_type="generation",
        name="openai-completion",
        model=model,
        input=[{"role": "user", "content": "Summarize: LangFuse is an open-source observability platform for LLM apps."}]
    ) as generation:
        prompt = "Summarize: LangFuse is an open-source observability platform for LLM apps."

        # Make the OpenAI API call
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )

        output = response.choices[0].message.content

        # Update the generation with output and usage details
        generation.update(
            output=output,
            usage_details={
                "input": response.usage.prompt_tokens,
                "output": response.usage.completion_tokens,
                "total": response.usage.total_tokens
            }
        )

        print(f"✅ Trace sent to LangFuse dashboard!")
        print(f"Model: {model}")
        print(f"User ID: {user_id}")
        print(f"Session ID: {session_id}")
        print(f"Input: {prompt}")
        print(f"Output: {output}")
        print(f"Token Usage: {response.usage.total_tokens} tokens (Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens})")

    span.update(output={"result": output})

# Ensure all traces are sent
langfuse.flush()

"""Run native Jev questions through Microsoft Agent Framework.

Usage: uv run 05_maf_jev.py
Configuration: JEV_ENDPOINT, JEV_DEPLOYMENT_NAME, and JEV_KEY in .env.
Uses the alpha TypeSafe extension and preserves native probabilities and usage.
"""

import asyncio
import json
import os
from time import perf_counter

from agent_framework import Agent
from agent_framework.exceptions import AgentException, ChatClientException
from agent_framework_typesafe import TypeSafeChatClient
from dotenv import load_dotenv
from rich import print_json
from typesafe_sdk import (
    AsyncTypeSafeClient,
    Choice,
    Noul,
    RetryPolicy,
    Score,
    SystemOneResponse,
    TypeSafeError,
)

load_dotenv()

JEV_ENDPOINT = os.environ["JEV_ENDPOINT"].rstrip("/")
DEPLOYMENT_NAME = os.environ["JEV_DEPLOYMENT_NAME"]
API_KEY = os.environ["JEV_KEY"]
TIMEOUT_SECONDS = 60


class DecisionAPIError(RuntimeError):
    """Raised when the API can't return a valid decision response."""


async def main() -> None:
    """Send the combined questions through MAF and print the typed response."""
    state = {
        "message": "Checkout is down again. This is my third request.",
        "customer_contact_count": 3,
    }
    questions = {
        "team": Choice(
            instructions="Which team should handle this request?",
            criteria={
                "billing": "Charges, invoices, and refunds",
                "engineering": "Bugs, errors, and outages",
                "support": "How-to and account questions",
            },
        ),
        "severity": Score(
            instructions="How severe is this issue?",
            criteria=["Cosmetic", "Minor", "Major", "Critical"],
        ),
        "repeat_contact": Noul(
            instructions="Has the customer contacted support before?"
        ),
    }
    # The SDK appends its own decision route to the service root.
    base_url = JEV_ENDPOINT.removesuffix("/v1/systemone")
    try:
        async with AsyncTypeSafeClient(
            api_key=API_KEY,
            model=DEPLOYMENT_NAME,
            base_url=base_url,
            timeout=TIMEOUT_SECONDS,
            retry=RetryPolicy(max_retries=0),
        ) as sdk_client:
            async with TypeSafeChatClient(
                async_client=sdk_client,
                model=DEPLOYMENT_NAME,
                base_url=base_url,
            ) as client:
                agent = Agent(client=client, name="JevAgent")
                started_at = perf_counter()
                response = await agent.run(
                    json.dumps(state), options={"response_format": questions}
                )
                latency_ms = (perf_counter() - started_at) * 1000
    except (TypeSafeError, ChatClientException, AgentException) as exc:
        raise DecisionAPIError(f"The Jev request failed: {exc}") from exc

    if not isinstance(response.value, SystemOneResponse):
        raise DecisionAPIError("The API didn't return a typed decision response.")
    if set(response.value.answers) != set(questions):
        raise DecisionAPIError("The API didn't return all three requested answers.")

    print_json(data=response.value.model_dump(mode="json"), indent=4)
    print(f"\nHTTP request latency: {latency_ms:.1f} ms")


if __name__ == "__main__":
    asyncio.run(main())

"""Run the combined Ollama Laya example using the MAF TypeSafe extension.

Usage: uv run 07_ollama.py
Configuration: OLLAMA_ENDPOINT and OLLAMA_DEPLOYMENT_NAME in .env.
Requires a native decision model such as Laya, not a chat-only Ollama model.
"""

import asyncio
import json
import os
from time import perf_counter
from uuid import uuid4

import httpx2
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

OLLAMA_ENDPOINT = os.environ["OLLAMA_ENDPOINT"].rstrip("/")
DEPLOYMENT_NAME = os.environ["OLLAMA_DEPLOYMENT_NAME"]
TIMEOUT_SECONDS = 120


class DecisionAPIError(RuntimeError):
    """Raised when Ollama can't return a valid native decision response."""


async def _route_decision_request(request: httpx2.Request) -> None:
    """Use the configured native endpoint and attach a correlation ID."""
    request.url = httpx2.URL(OLLAMA_ENDPOINT)
    request.headers["x-request-id"] = str(uuid4())


async def _map_request_id(response: httpx2.Response) -> None:
    """Provide tracing metadata expected by the TypeSafe MAF extension."""
    if "x-typesafe-request-id" not in response.headers:
        response.headers["x-typesafe-request-id"] = (
            response.headers.get("x-request-id")
            or response.request.headers["x-request-id"]
        )


async def main() -> None:
    """Run the same three typed questions locally through the MAF extension."""
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
    base_url = str(httpx2.URL(OLLAMA_ENDPOINT).copy_with(path="/", query=None))
    try:
        async with AsyncTypeSafeClient(
            # The SDK requires a key argument; local Ollama does not use this value.
            api_key="ollama-local",
            model=DEPLOYMENT_NAME,
            base_url=base_url,
            timeout=TIMEOUT_SECONDS,
            retry=RetryPolicy(max_retries=0),
            http_client=httpx2.AsyncClient(
                timeout=TIMEOUT_SECONDS,
                event_hooks={
                    "request": [_route_decision_request],
                    "response": [_map_request_id],
                },
            ),
        ) as sdk_client:
            async with TypeSafeChatClient(
                async_client=sdk_client, model=DEPLOYMENT_NAME, base_url=base_url
            ) as client:
                agent = Agent(client=client, name="OllamaDecisionAgent")
                started_at = perf_counter()
                response = await agent.run(
                    json.dumps(state), options={"response_format": questions}
                )
                latency_ms = (perf_counter() - started_at) * 1000
    except (
        TypeSafeError,
        ChatClientException,
        AgentException,
        httpx2.HTTPError,
    ) as exc:
        raise DecisionAPIError(f"The Ollama decision request failed: {exc}") from exc

    if not isinstance(response.value, SystemOneResponse):
        raise DecisionAPIError("Ollama didn't return a typed decision response.")
    if set(response.value.answers) != set(questions):
        raise DecisionAPIError("Ollama didn't return all three requested answers.")
    print_json(data=response.value.model_dump(mode="json"), indent=4)
    print(f"\nHTTP request latency: {latency_ms:.1f} ms")


if __name__ == "__main__":
    asyncio.run(main())

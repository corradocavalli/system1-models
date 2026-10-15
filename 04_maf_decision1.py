"""Run native Azure Decision-1 questions through Microsoft Agent Framework.

Usage: uv run 04_maf_decision1.py
Configuration: AZURE_ENDPOINT and DEPLOYMENT_NAME in .env; Azure Entra credentials.
Uses the alpha TypeSafe extension and preserves native probabilities and usage.
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
from azure.identity import DefaultAzureCredential
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

AZURE_ENDPOINT = os.environ["AZURE_ENDPOINT"].rstrip("/")
DEPLOYMENT_NAME = os.environ["DEPLOYMENT_NAME"]
TIMEOUT_SECONDS = 60
TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"


class DecisionAPIError(RuntimeError):
    """Raised when the API can't return a valid decision response."""


async def _route_decision_request(request: httpx2.Request) -> None:
    """Replace the SDK's fixed /v1/systemone route with the Azure endpoint."""
    request.url = httpx2.URL(AZURE_ENDPOINT)
    request.headers["x-ms-client-request-id"] = str(uuid4())


async def _map_request_id(response: httpx2.Response) -> None:
    """Supply SDK tracing metadata from Azure or the client correlation ID."""
    if "x-typesafe-request-id" not in response.headers:
        response.headers["x-typesafe-request-id"] = (
            response.headers.get("apim-request-id")
            or response.headers.get("x-ms-request-id")
            or response.headers.get("x-request-id")
            or response.request.headers["x-ms-client-request-id"]
        )


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
    with DefaultAzureCredential() as credential:
        access_token = credential.get_token(TOKEN_SCOPE).token

    base_url = str(httpx2.URL(AZURE_ENDPOINT).copy_with(path="/", query=None))
    # The hooks adapt Azure routing and tracing headers to the TypeSafe SDK contract.
    try:
        async with AsyncTypeSafeClient(
            api_key=access_token,
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
                async_client=sdk_client,
                model=DEPLOYMENT_NAME,
                base_url=base_url,
            ) as client:
                agent = Agent(client=client, name="Decision1Agent")
                started_at = perf_counter()
                response = await agent.run(
                    json.dumps(state), options={"response_format": questions}
                )
                latency_ms = (perf_counter() - started_at) * 1000
    except (TypeSafeError, ChatClientException, AgentException) as exc:
        raise DecisionAPIError(f"The Decision-1 request failed: {exc}") from exc

    if not isinstance(response.value, SystemOneResponse):
        raise DecisionAPIError("The API didn't return a typed decision response.")
    if set(response.value.answers) != set(questions):
        raise DecisionAPIError("The API didn't return all three requested answers.")

    print_json(data=response.value.model_dump(mode="json"), indent=4)
    print(f"\nHTTP request latency: {latency_ms:.1f} ms")


if __name__ == "__main__":
    asyncio.run(main())

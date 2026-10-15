"""Run the combined decision example using the OpenAI Decisions API.

Usage: uv run 06_openai_decision.py
Configuration: OPENAI_ENDPOINT, OPENAI_DEPLOYMENT_NAME, and OPENAI_KEY in .env.
Prints native choice, score, and predicate answers with token usage and latency.
"""

import json
import os
from time import perf_counter

from dotenv import load_dotenv
from openai import APIError, OpenAI
from openai.types.decision import Decision
from rich import print_json

load_dotenv()

OPENAI_ENDPOINT = os.environ["OPENAI_ENDPOINT"].rstrip("/")
DEPLOYMENT_NAME = os.environ["OPENAI_DEPLOYMENT_NAME"]
API_KEY = os.environ["OPENAI_KEY"]
TIMEOUT_SECONDS = 60


class DecisionAPIError(RuntimeError):
    """Raised when the API can't return a valid decision response."""


def main() -> None:
    """Send the combined questions and print the native decision JSON."""
    state = {
        "message": "Checkout is down again. This is my third request.",
        "customer_contact_count": 3,
    }
    try:
        with OpenAI(
            api_key=API_KEY,
            base_url=OPENAI_ENDPOINT.removesuffix("/decisions") + "/",
            timeout=TIMEOUT_SECONDS,
            max_retries=0,
        ) as client:
            started_at = perf_counter()
            decision = client.decisions.create(
                model=DEPLOYMENT_NAME,
                input=json.dumps(state),
                questions=[
                    {
                        "type": "choice",
                        "name": "team",
                        "instructions": "Which team should handle this request?",
                        "choices": [
                            {
                                "value": "billing",
                                "description": "Charges, invoices, and refunds",
                            },
                            {
                                "value": "engineering",
                                "description": "Bugs, errors, and outages",
                            },
                            {
                                "value": "support",
                                "description": "How-to and account questions",
                            },
                        ],
                    },
                    {
                        "type": "score",
                        "name": "severity",
                        "instructions": "How severe is this issue?",
                        "levels": [
                            {"label": "Cosmetic"},
                            {"label": "Minor"},
                            {"label": "Major"},
                            {"label": "Critical"},
                        ],
                    },
                    # OpenAI calls its native yes/no question type a predicate.
                    {
                        "type": "predicate",
                        "name": "repeat_contact",
                        "instructions": "Has the customer contacted support before?",
                    },
                ],
            )
            latency_ms = (perf_counter() - started_at) * 1000
    except APIError as exc:
        raise DecisionAPIError(f"The Decisions API request failed: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise DecisionAPIError("The API returned invalid JSON.") from exc

    if not isinstance(decision, Decision):
        raise DecisionAPIError("The API didn't return a valid decision response.")

    print_json(data=decision.model_dump(mode="json"), indent=4)
    print(f"\nHTTP request latency: {latency_ms:.1f} ms")


if __name__ == "__main__":
    main()

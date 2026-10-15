"""Ask Azure Decision-1 choice, score, and yes/no questions in one REST request.

Usage: uv run 02_decision1_combined.py
Configuration: AZURE_ENDPOINT and DEPLOYMENT_NAME in .env; Azure Entra credentials.
"""

import os
from time import perf_counter

import requests
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from rich import print_json

load_dotenv()

AZURE_ENDPOINT = os.environ["AZURE_ENDPOINT"].rstrip("/")
DEPLOYMENT_NAME = os.environ["DEPLOYMENT_NAME"]
TIMEOUT_SECONDS = 60
TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"


class DecisionAPIError(RuntimeError):
    """Raised when the API can't return a valid response."""


def main() -> None:
    """Send one combined decision request and print its JSON response."""
    body = {
        "model": DEPLOYMENT_NAME,
        "state": {
            "message": "Checkout is down again. This is my third request.",
            "customer_contact_count": 3,
        },
        "questions": {
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this request?",
                "criteria": {
                    "billing": "Charges, invoices, and refunds",
                    "engineering": "Bugs, errors, and outages",
                    "support": "How-to and account questions",
                },
            },
            "severity": {
                "type": "score",
                "instructions": "How severe is this issue?",
                "criteria": ["Cosmetic", "Minor", "Major", "Critical"],
            },
            # Noul returns a probability in [0, 1] for a yes/no question.
            "repeat_contact": {
                "type": "noul",
                "instructions": "Has the customer contacted support before?",
            },
        },
    }
    with DefaultAzureCredential() as credential:
        access_token = credential.get_token(TOKEN_SCOPE).token

    try:
        started_at = perf_counter()
        response = requests.post(
            AZURE_ENDPOINT,
            json=body,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Accept": "application/json",
            },
            timeout=TIMEOUT_SECONDS,
        )
        latency_ms = (perf_counter() - started_at) * 1000
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise DecisionAPIError(
            f"The API returned HTTP {exc.response.status_code}."
        ) from exc
    except requests.RequestException as exc:
        raise DecisionAPIError(f"Couldn't reach the API: {exc}.") from exc

    try:
        payload = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise DecisionAPIError("The API returned invalid JSON.") from exc
    print_json(data=payload, indent=4)
    print(f"\nHTTP request latency: {latency_ms:.1f} ms")


if __name__ == "__main__":
    main()

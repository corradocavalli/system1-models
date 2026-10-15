"""Classify support requests with Azure Decision-1 and report accuracy and latency.

Usage: uv run 01_decision1.py
Configuration: AZURE_ENDPOINT and DEPLOYMENT_NAME in .env; Azure Entra credentials.
"""

import os
from collections.abc import Mapping
from statistics import median
from time import perf_counter
from typing import Any

import requests
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
from rich import print_json

load_dotenv()

OPTIONS = {
    "billing": "Charges, invoices, refunds, or subscription payments",
    "technical": "Software errors, bugs, or integration failures",
    "account": "Sign-in, password, or account-access problems",
    "other": "Other issues not covered by the above categories",
}

EXAMPLES = [
    ("I was charged twice.", "billing"),
    ("The integration crashes during checkout.", "technical"),
    ("I cannot sign in after resetting my password.", "account"),
    ("I have an error when I launch MS Teams.", "technical"),
    ("I need a holiday.", "other"),
]

AZURE_ENDPOINT = os.environ["AZURE_ENDPOINT"].rstrip("/")
DEPLOYMENT_NAME = os.environ["DEPLOYMENT_NAME"]
TIMEOUT_SECONDS = 60
TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"
CREDENTIAL = DefaultAzureCredential()


class DecisionAPIError(RuntimeError):
    """Raised when the API can't return a valid classification."""


def read_choice(
    payload: Any,
    options: Mapping[str, str],
) -> str:
    """Extract a native choice answer and reject unknown categories."""
    if not isinstance(payload, dict):
        raise DecisionAPIError("The API returned an invalid response.")

    answers = payload.get("answers")
    if not isinstance(answers, dict):
        raise DecisionAPIError("The API returned no answers object.")

    answer = answers.get("team")
    if not isinstance(answer, dict) or answer.get("type") != "choice":
        raise DecisionAPIError("The API returned an invalid answer.")

    choice = answer.get("choice")
    if not isinstance(choice, str) or choice not in options:
        raise DecisionAPIError(f"The API selected an unsupported team: {choice!r}.")
    return choice


def predict(
    text: str,
    options: Mapping[str, str],
) -> tuple[str, float]:
    """Print the native response and return its choice and HTTP latency in ms."""
    if not text.strip():
        raise ValueError("text must not be empty.")
    if len(options) < 2:
        raise ValueError("options must contain at least two choices.")

    body = {
        "model": DEPLOYMENT_NAME,
        "state": text,
        "questions": {
            "team": {
                "type": "choice",
                "instructions": (
                    "Which team should handle this customer support request? Select exactly one team based on theprimary problem."
                ),
                "criteria": dict(options),
            }
        },
    }
    access_token = CREDENTIAL.get_token(TOKEN_SCOPE).token

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
        print_json(data=payload, indent=4)
    except requests.exceptions.JSONDecodeError as exc:
        raise DecisionAPIError("The API returned invalid JSON.") from exc
    return read_choice(payload, options), latency_ms


def main() -> None:
    """Compare predictions with the example labels and summarize results."""
    correct = 0
    latencies_ms = []
    rows: list[str] = []
    request_width = max(len("Request"), *(len(text) for text, _ in EXAMPLES))
    header = (
        f"{'Request':<{request_width}} | {'Expected':<10} | {'Predicted':<10} | "
        f"{'Result':<6} | {'HTTP (ms)':>12}"
    )
    for text, expected in EXAMPLES:
        predicted, latency_ms = predict(text, OPTIONS)

        correct += int(predicted == expected)
        latencies_ms.append(latency_ms)
        result = "PASS" if predicted == expected else "FAIL"
        rows.append(
            f"{text:<{request_width}} | {expected:<10} | {predicted:<10} | "
            f"{result:<6} | {latency_ms:>12.1f}"
        )

    print(f"\n{header}")
    print("-" * len(header))
    print("\n".join(rows))

    accuracy = correct / len(EXAMPLES)
    print(f"\nAccuracy: {accuracy:.1%}")
    print(f"Median HTTP request latency: {median(latencies_ms):.1f} ms")


if __name__ == "__main__":
    main()

"""
classify_text_to_json.py

Purpose:
- Input plain text immunisation question or transcript.
- Use Azure OpenAI gpt-4o-mini to classify the immunisation query.
- Output a JSON file containing:
  1. the original input text
  2. the AI classification result

Expected category/.env in the same folder as this script:

CLASSIFIER_AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/
CLASSIFIER_AZURE_OPENAI_API_KEY=YOUR-API-KEY
CLASSIFIER_AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini-medical-chatbot
CLASSIFIER_AZURE_OPENAI_API_VERSION=2024-10-21

Example usage:

1. Classify text directly:
   python classify_text_to_json.py --text "Can a 50-year-old still receive Tdap if they missed the 45-year vaccination?" --output result_text.json

2. Classify text from a .txt file:
   python classify_text_to_json.py --text-file question.txt --output result_text.json
"""

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from openai import AzureOpenAI


# ============================================================
# category/.env loading for the classification agent only
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent


def load_env_file(env_path: Path) -> None:
    """
    Minimal .env loader for category/.env.

    This script intentionally uses CLASSIFIER_* variable names so it will not
    conflict with the root project .env used by the RAG answer model.
    """
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        # Do not overwrite environment variables explicitly set by Colab/shell.
        os.environ.setdefault(key, value)


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing environment variable: {name}. "
            f"Please set it in category/.env next to classify_text_to_json.py."
        )
    return value


load_env_file(SCRIPT_DIR / ".env")

# Internal variable names are kept as AZURE_OPENAI_* to minimise changes to the
# original classification logic. They are populated from CLASSIFIER_* env vars.
AZURE_OPENAI_ENDPOINT = require_env("CLASSIFIER_AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = require_env("CLASSIFIER_AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = require_env("CLASSIFIER_AZURE_OPENAI_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("CLASSIFIER_AZURE_OPENAI_API_VERSION", "2024-10-21").strip()

if "/openai/deployments/" in AZURE_OPENAI_ENDPOINT:
    raise RuntimeError(
        "CLASSIFIER_AZURE_OPENAI_ENDPOINT should be the Azure resource root URL, "
        "for example https://ai-team-09-hack.cognitiveservices.azure.com/ . "
        "Do not use the full /openai/deployments/.../chat/completions URL."
    )


MAIN_CATEGORIES = [
    "Schedule",
    "Eligibility / Funding",
    "Catch-up",
    "Contraindication / Precaution",
    "Special Population",
    "Administration",
    "AEFI / Safety",
    "Anaphylaxis / Emergency",
    "Storage / Cold Chain",
    "Consent / Documentation",
    "Disease-specific",
    "Other / Unclear",
]

CLINICAL_SCENARIOS = [
    "Missed dose",
    "Delayed vaccination",
    "Adult booster",
    "Childhood schedule",
    "Pregnancy",
    "Breastfeeding",
    "Immunocompromised",
    "Immunosuppressant therapy",
    "Medicine interaction",
    "Previous adverse reaction",
    "Allergy",
    "Post-exposure",
    "Travel",
    "Funding eligibility",
    "Catch-up after overseas vaccination",
    "Co-administration",
    "Spacing interval",
    "Storage temperature issue",
    "Consent issue",
    "Documentation issue",
    "Other / Unclear",
]

TARGET_POPULATIONS = [
    "Infant",
    "Child",
    "Adolescent",
    "Adult",
    "Older adult",
    "Pregnant",
    "Breastfeeding",
    "Immunocompromised",
    "Healthcare worker",
    "Traveller",
    "Unknown",
]

RISK_LEVELS = ["Low", "Medium", "High", "Unknown"]
CONFIDENCE_LEVELS = ["Low", "Medium", "High", "Unknown"]

CALLER_TYPES = [
    "Healthcare professional",
    "Public caller",
    "Parent or caregiver",
    "Unknown",
]

VACCINE_NOISE_WORDS = {
    "NAME",
    "name",
    "the",
    "a",
    "an",
    "vaccine",
    "vaccination",
    "dose",
    "booster",
    "patient",
    "person",
    "someone",
    "somebody",
    "adult",
    "child",
    "infant",
    "caller",
    "customer",
    "agent",
    "unknown",
    "Unknown",
}


# ============================================================
# PII redaction
# ============================================================

PII_REDACTION_CONFIG = {
    "PII": {
        "RedactionEntitiesRequested": [
            "BANK_ACCOUNT_NUMBER",
            "BANK_ROUTING",
            "CREDIT_DEBIT_NUMBER",
            "CREDIT_DEBIT_CVV",
            "CREDIT_DEBIT_EXPIRY",
            "INTERNATIONAL_BANK_ACCOUNT_NUMBER",
            "PIN",
            "SWIFT_CODE",
            "CA_HEALTH_NUMBER",
            "UK_NATIONAL_HEALTH_SERVICE_NUMBER",
            "CA_SOCIAL_INSURANCE_NUMBER",
            "SSN",
            "UK_NATIONAL_INSURANCE_NUMBER",
            "PASSPORT_NUMBER",
            "DRIVER_ID",
            "IN_AADHAAR",
            "NAME",
            "EMAIL",
            "PHONE",
            "ADDRESS",
            "US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER",
            "UK_UNIQUE_TAXPAYER_REFERENCE_NUMBER",
            "IN_PERMANENT_ACCOUNT_NUMBER",
            "IN_NREGA",
            "AWS_ACCESS_KEY",
            "AWS_SECRET_KEY",
            "IP_ADDRESS",
            "MAC_ADDRESS",
            "PASSWORD",
            "URL",
            "USERNAME",
            "LICENSE_PLATE",
            "VEHICLE_IDENTIFICATION_NUMBER",
            "IN_VOTER_NUMBER",
            "CUSTOMER_DISPLAY_NAME",
            "ATTACHMENT_NAME",
        ],
        "RedactionMaskMode": "FixedMask",
        "RedactionMask": "**",
    }
}


PII_REGEX_PATTERNS = [
    ("AWS_ACCESS_KEY", r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    ("AWS_SECRET_KEY", r"\b[A-Za-z0-9/+=]{40}\b"),
    ("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    ("URL", r"\bhttps?://[^\s<>()]+|\bwww\.[^\s<>()]+"),
    ("IP_ADDRESS", r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
    ("MAC_ADDRESS", r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"),
    ("CREDIT_DEBIT_NUMBER", r"\b(?:\d[ -]*?){13,19}\b"),
    ("CREDIT_DEBIT_CVV", r"\b(?:CVV|CVC|security code)\s*[:=]?\s*\d{3,4}\b"),
    ("CREDIT_DEBIT_EXPIRY", r"\b(?:0[1-9]|1[0-2])\s*/\s*(?:\d{2}|\d{4})\b"),
    ("INTERNATIONAL_BANK_ACCOUNT_NUMBER", r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    ("SWIFT_CODE", r"\b[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b"),
    ("SSN", r"\b\d{3}-\d{2}-\d{4}\b"),
    ("UK_NATIONAL_INSURANCE_NUMBER", r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b"),
    ("UK_NATIONAL_HEALTH_SERVICE_NUMBER", r"\b\d{3}\s?\d{3}\s?\d{4}\b"),
    ("PASSPORT_NUMBER", r"\b(?:passport|passport no\.?|passport number)\s*[:=]?\s*[A-Z0-9]{6,12}\b"),
    ("DRIVER_ID", r"\b(?:driver licence|driver license|driver id)\s*[:=]?\s*[A-Z0-9-]{5,20}\b"),
    ("PHONE", r"\b(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}\b"),
    ("PASSWORD", r"\b(?:password|passwd|pwd)\s*[:=]\s*[^\s,;]+"),
    ("USERNAME", r"\b(?:username|user name|login)\s*[:=]\s*[A-Za-z0-9._-]+"),
    ("PIN", r"\b(?:PIN|pin)\s*[:=]?\s*\d{4,8}\b"),
    ("LICENSE_PLATE", r"\b(?:licen[cs]e plate|rego|registration plate)\s*[:=]?\s*[A-Z0-9-]{2,10}\b"),
    ("VEHICLE_IDENTIFICATION_NUMBER", r"\b[A-HJ-NPR-Z0-9]{17}\b"),
    ("ADDRESS", r"\b\d{1,5}\s+[A-Za-z0-9.' -]+(?:Street|St|Road|Rd|Avenue|Ave|Drive|Dr|Lane|Ln|Way|Place|Pl|Crescent|Cres|Boulevard|Blvd)\b"),
    ("NAME", r"\b(?:my name is|name is|patient name is|caller is)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b"),
]


CASE_SENSITIVE_PII_ENTITIES = {
    "SWIFT_CODE",
    "AWS_ACCESS_KEY",
    "INTERNATIONAL_BANK_ACCOUNT_NUMBER",
    "UK_NATIONAL_INSURANCE_NUMBER",
    "VEHICLE_IDENTIFICATION_NUMBER",
}


def anonymize_text(text: str):
    """
    Locally redact common PII patterns before classification and retrieval.

    Returns:
      redacted_text: text with fixed privacy mask, such as **
      redactions: list of detected entity summaries

    Note:
      Regex redaction is a lightweight safeguard for the prototype.
      Production systems should use a managed PII service such as Azure AI Language PII.
    """
    redactions = []
    redacted_text = text

    for entity_type, pattern in PII_REGEX_PATTERNS:
        if entity_type not in PII_REDACTION_CONFIG["PII"]["RedactionEntitiesRequested"]:
            continue

        mask = PII_REDACTION_CONFIG["PII"].get("RedactionMask", "**")

        def _replace(match):
            matched_text = match.group(0)
            redactions.append(
                {
                    "entity_type": entity_type,
                    "mask": mask,
                    "start": match.start(),
                    "end": match.end(),
                }
            )
            return mask

        flags = 0 if entity_type in CASE_SENSITIVE_PII_ENTITIES else re.IGNORECASE
        redacted_text = re.sub(pattern, _replace, redacted_text, flags=flags)

    return redacted_text, redactions



SYSTEM_PROMPT = f"""
You are an immunisation call classification assistant for New Zealand immunisation advisor calls.

Your role is NOT to answer the clinical question.
Your role is to classify the caller's immunisation query for:
- reporting
- workflow triage
- RAG routing
- handbook section retrieval

Return ONLY valid JSON.
Do not return markdown.
Do not explain your reasoning outside the JSON.

Core principles:
- Classify based on the caller's original question, not the advisor's final answer.
- Use only information explicitly present in the input text.
- Do not invent patient details, medical history, age, pregnancy status, vaccine history, eligibility, or funding status.
- Do not provide clinical advice.
- Do not include personal information.
- If information is missing, use "Unknown" or [].
- Prefer the New Zealand Immunisation Handbook as the main guidance source when the query can be answered by the Handbook.
- If the question concerns vaccine funding, product information, or medicine approval, include PHARMAC or Medsafe as an additional source as appropriate.

Allowed main_category:
{MAIN_CATEGORIES}

Allowed clinical_scenario:
{CLINICAL_SCENARIOS}

Allowed target_population:
{TARGET_POPULATIONS}

Allowed risk_level:
{RISK_LEVELS}

Classification priority rules:
1. If the caller asks about a missed, delayed, overdue, late, or past scheduled vaccine, choose "Catch-up" as main_category.
2. If the caller mainly asks whether someone can receive a vaccine based on age, medical condition, pregnancy, immune status, vaccine history, or previous infection, choose "Eligibility / Funding" unless the main issue is clearly catch-up.
3. If the caller mainly asks when a vaccine should be given, choose "Schedule".
4. If the caller asks whether a vaccine should not be given because of allergy, adverse reaction, current illness, pregnancy, immune suppression, medicine interaction, or another safety issue, choose "Contraindication / Precaution".
5. If the caller asks about vaccine preparation, injection site, route, dosage, spacing, storage, cold chain, or administration process, choose "Administration" or "Storage / Cold Chain".
6. If the caller asks about an adverse event after vaccination, choose "AEFI / Safety".
7. If the caller asks about anaphylaxis, collapse, severe allergic reaction, or emergency response, choose "Anaphylaxis / Emergency".
8. If the caller asks about consent, records, AIR, documentation, or proof of vaccination, choose "Consent / Documentation".
9. Funding should only be the main_category if the caller's main question is about whether the vaccine is funded, free, covered, or eligible for public funding.
10. If multiple categories are possible, choose the category that best represents the caller's main operational need.

Vaccine and disease extraction rules:
- Extract ONLY vaccines, vaccine-preventable diseases, vaccine abbreviations, and vaccine brand names.
- Do NOT include non-vaccine medicines, immunosuppressants, biologics, injections, or general treatments in vaccine_or_disease.
- If a non-vaccine medicine, biologic, immunosuppressant, injection, or treatment is clinically relevant, put it in medicine_or_treatment instead.
- Dynamically extract vaccine names, disease names, vaccine abbreviations, and vaccine brand names from the text.
- Keep meaningful vaccine or disease names such as Tdap, MMR, HPV, MenB, PCV13, influenza, varicella, pertussis, tetanus, COVID-19, Boostrix, Bexsero, Shingrix.
- Normalize common vaccine names:
  - "flu vaccine" -> "Influenza vaccine"
  - "flu jab" -> "Influenza vaccine"
  - "flu" when clearly referring to vaccination -> "Influenza vaccine"
  - "COVID vaccine" -> "COVID-19 vaccine"
- Remove noise words such as NAME, vaccine, vaccination, dose, booster, patient, person, someone, somebody, adult, child, caller, agent.
- If no vaccine or vaccine-preventable disease is mentioned, use ["Unknown"].

Medicine or treatment extraction rules:
- Use medicine_or_treatment for non-vaccine medicines, biologics, immunosuppressants, injections, or treatments mentioned in the text.
- Do not place these items in vaccine_or_disease.
- If no medicine or treatment is mentioned, use ["Unknown"].

Target population rules:
- If the text mentions a person aged 18 years or older, include "Adult".
- If the text mentions 65 years or older, include both "Adult" and "Older adult".
- If the text mentions a baby or infant, include "Infant".
- If the text mentions a child but no specific age, include "Child".
- If pregnancy is mentioned, include "Pregnant".
- If breastfeeding is mentioned, include "Breastfeeding".
- If immunosuppression, transplant, chemotherapy, HIV, renal dialysis, asplenia, immune deficiency, biologic therapy, or immunosuppressant medicine is mentioned, include "Immunocompromised".
- If the caller is a nurse, doctor, pharmacist, vaccinator, clinic staff, or health professional, this is caller_type, not target_population.
- If the target person is not clear, use ["Unknown"].

Caller type rules:
- If the caller says they are calling from a clinic, medical centre, pharmacy, hospital, GP practice, healthcare organisation, or uses language suggesting they are asking on behalf of a patient, use "Healthcare professional".
- If the caller asks about "my child", "my baby", "my son", "my daughter", or another dependent, use "Parent or caregiver".
- If the caller asks about themselves and no healthcare role is indicated, use "Public caller".
- If unclear, use "Unknown".

Risk level rules:
- "High": possible anaphylaxis, emergency, severe adverse event, serious contraindication, immunocompromised live vaccine concern, pregnancy live vaccine concern, or urgent safety issue.
- "Medium": incomplete clinical details, uncertain contraindication, special population, immunosuppression, medicine interaction, prior adverse reaction, cold chain breach, or complex eligibility/funding issue.
- "Low": routine schedule, catch-up, general eligibility, routine adult booster, documentation, simple spacing question, or simple clarification.
- "Unknown": insufficient information.
- If the immunisation query is classifiable, avoid "Unknown" risk_level unless there is truly not enough information to assess even basic triage risk.
- Use "Medium" for immunosuppression, medicine interaction, unclear contraindication, or special population questions.
- Use "Low" for routine catch-up, routine schedule, documentation, and simple co-administration questions without special risk factors.

Source selection rules:
- Use "Handbook" for schedule, catch-up, contraindications, special populations, administration, AEFI, anaphylaxis, and general clinical guidance.
- Use "PHARMAC" for funding, eligibility for publicly funded vaccines, and Pharmaceutical Schedule confirmation.
- Use "Medsafe" for product data sheets, approved indications, vaccine components, contraindications from data sheets, and medicine safety information.
- Use "IMAC website" for practical FAQs, factsheets, or advisor-facing summaries.
- Do not output ["Unknown"] for source_needed if the text contains an identifiable immunisation question.
- If the category is identified but the exact source is uncertain, use ["Handbook", "IMAC website"].
- Use ["Unknown"] only when the text does not contain a classifiable immunisation query.

Handbook search term rules:
Generate terms that would help retrieve the correct section of the Immunisation Handbook.
Include:
- vaccine or disease name
- scenario
- age or life stage
- key clinical issue
- special population if relevant
- medicine or treatment if relevant

Output JSON schema:
{{
  "extracted_question": "short rewritten clinical question from the caller's perspective",
  "main_category": "one allowed main_category",
  "secondary_category": "short free-text subtype",
  "vaccine_or_disease": ["dynamically extracted vaccine or disease names only"],
  "medicine_or_treatment": ["non-vaccine medicines, biologics, immunosuppressants, injections, or treatments mentioned, or Unknown"],
  "target_population": ["one or more allowed target_population values"],
  "caller_type": "Healthcare professional / Public caller / Parent or caregiver / Unknown",
  "clinical_scenario": ["one or more allowed clinical_scenario values"],
  "age_or_life_stage": ["any age, gestation, or life-stage information explicitly mentioned"],
  "risk_level": "Low / Medium / High / Unknown",
  "source_needed": ["Handbook", "IMAC website", "Medsafe", "PHARMAC", "Unknown"],
  "handbook_search_terms": ["specific search terms for RAG retrieval"],
  "needs_rag": true,
  "confidence": "Low / Medium / High",
  "evidence_from_transcript": ["short exact phrases from the input text that justify the classification"]
}}

Output validation rules:
- Return exactly one JSON object.
- Use only allowed values for main_category, target_population, clinical_scenario, and risk_level.
- Do not include markdown, comments, or extra text.
- evidence_from_transcript must contain short exact phrases from the input text.
- If a field is unknown, use "Unknown" for a string field or ["Unknown"] for a list field.
- Do not include private names, addresses, phone numbers, emails, or other personal information.

Now classify the input text.
"""


def create_azure_client() -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
    )


def save_json(path: Path, data: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_text_file(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Text path is not a file: {path}")
    return path.read_text(encoding="utf-8").strip()


def ensure_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value).strip()]


def clean_list_against_allowed(values: List[str], allowed: List[str], default: str) -> List[str]:
    cleaned = []
    for value in values:
        if value in allowed:
            cleaned.append(value)
    return sorted(set(cleaned)) if cleaned else [default]


def clean_string_against_allowed(value: Any, allowed: List[str], default: str) -> str:
    value = str(value).strip() if value is not None else ""
    return value if value in allowed else default


def clean_vaccine_or_disease(values: List[str]) -> List[str]:
    cleaned = []
    noise_words_lower = {word.lower() for word in VACCINE_NOISE_WORDS}

    for value in values:
        v = value.strip().strip(".,;:()[]{}")
        if not v:
            continue
        if v.lower() in noise_words_lower:
            continue
        cleaned.append(v)

    return sorted(set(cleaned)) if cleaned else ["Unknown"]


def normalize_classification(raw: Dict[str, Any]) -> Dict[str, Any]:
    main_category = clean_string_against_allowed(
        raw.get("main_category"),
        MAIN_CATEGORIES,
        "Other / Unclear",
    )

    risk_level = clean_string_against_allowed(
        raw.get("risk_level"),
        RISK_LEVELS,
        "Unknown",
    )

    confidence = clean_string_against_allowed(
        raw.get("confidence"),
        CONFIDENCE_LEVELS,
        "Unknown",
    )

    caller_type = clean_string_against_allowed(
        raw.get("caller_type"),
        CALLER_TYPES,
        "Unknown",
    )

    clinical_scenario = clean_list_against_allowed(
        ensure_list(raw.get("clinical_scenario")),
        CLINICAL_SCENARIOS,
        "Other / Unclear",
    )

    target_population = clean_list_against_allowed(
        ensure_list(raw.get("target_population")),
        TARGET_POPULATIONS,
        "Unknown",
    )

    vaccine_or_disease = clean_vaccine_or_disease(
        ensure_list(raw.get("vaccine_or_disease"))
    )

    medicine_or_treatment = ensure_list(raw.get("medicine_or_treatment")) or ["Unknown"]
    source_needed = ensure_list(raw.get("source_needed")) or ["Handbook", "IMAC website"]

    return {
        "extracted_question": str(raw.get("extracted_question", "")).strip(),
        "main_category": main_category,
        "secondary_category": str(raw.get("secondary_category", "")).strip(),
        "vaccine_or_disease": vaccine_or_disease,
        "medicine_or_treatment": medicine_or_treatment,
        "target_population": target_population,
        "caller_type": caller_type,
        "clinical_scenario": clinical_scenario,
        "age_or_life_stage": ensure_list(raw.get("age_or_life_stage")) or ["Unknown"],
        "risk_level": risk_level,
        "source_needed": source_needed,
        "handbook_search_terms": ensure_list(raw.get("handbook_search_terms")),
        "needs_rag": bool(raw.get("needs_rag", True)),
        "confidence": confidence,
        "evidence_from_transcript": ensure_list(raw.get("evidence_from_transcript")),
    }


def classify_text_with_mini(input_text: str, client: AzureOpenAI) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": input_text},
        ],
    )

    content = response.choices[0].message.content
    raw = json.loads(content)
    return normalize_classification(raw)


def build_output_json(
    original_text: str,
    classification: Dict[str, Any],
    text_id: str,
) -> Dict[str, Any]:
    return {
        "input_type": "plain_text",
        "text_id": text_id,
        "original_text": original_text,
        "AIClassification": {
            "model": AZURE_OPENAI_DEPLOYMENT,
            "api_version": AZURE_OPENAI_API_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "classification": classification,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify plain text immunisation questions and output original text plus classification as JSON."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text",
        help="Plain text immunisation question or transcript to classify.",
    )
    input_group.add_argument(
        "--text-file",
        help="Path to a .txt file containing the plain text to classify.",
    )

    parser.add_argument(
        "--output",
        default="text_classification_result.json",
        help="Output JSON file path. Default: text_classification_result.json",
    )

    parser.add_argument(
        "--text-id",
        default=None,
        help="Optional ID for this text input. If omitted, uses manual_input or the text file name.",
    )

    args = parser.parse_args()

    if args.text:
        original_text = args.text.strip()
        text_id = args.text_id or "manual_input"
    else:
        text_file_path = Path(args.text_file)
        original_text = read_text_file(text_file_path)
        text_id = args.text_id or text_file_path.stem

    if not original_text:
        raise ValueError("Input text is empty.")

    print("Running mode: plain text classification")
    print(f"Azure OpenAI classifier deployment: {AZURE_OPENAI_DEPLOYMENT}")

    redacted_text, redactions = anonymize_text(original_text)

    client = create_azure_client()
    classification = classify_text_with_mini(redacted_text, client)
    classification["pii_redaction"] = {
        "enabled": True,
        "redaction_config": PII_REDACTION_CONFIG,
        "redactions_detected": redactions,
        "redacted_input_text": redacted_text,
    }

    output_data = build_output_json(
        original_text=original_text,
        classification=classification,
        text_id=text_id,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_path, output_data)

    print(f"Saved result to: {output_path}")
    print(json.dumps(output_data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

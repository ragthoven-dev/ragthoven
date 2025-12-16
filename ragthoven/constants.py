PROMPT_LOGGING = """
Logging on first data:
====================== system prompt ======================
{sprompt}
====================== user prompt ======================
{uprompt}
====================== model response ======================
{pres}
"""

DEFAULT_BATCH_SIZE = 100

LLM_OVERRIDE_KEYS = ["model", "base_url", "temperature"]

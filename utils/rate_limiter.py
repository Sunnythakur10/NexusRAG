from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
import logging

# Configure basic logging so you can see when the pipeline is backing off
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_retry(retry_state):
    """Logs a warning to the terminal whenever Groq rate limits you."""
    logger.warning(
        f"[Rate Limit Hit] Pausing execution... Retrying in {retry_state.next_action.sleep} seconds. "
        f"(Attempt {retry_state.attempt_number})"
    )

def with_exponential_backoff():
    """
    Enterprise rate-limit handler.
    - multiplier=2: The base wait time.
    - min=2: Minimum wait of 2 seconds.
    - max=60: Never wait longer than 60 seconds per attempt.
    - stop_after_attempt(10): Give up and fail if Groq is down for 10 straight tries.
    """
    return retry(
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(10),
        before_sleep=log_retry,
        reraise=True
    )
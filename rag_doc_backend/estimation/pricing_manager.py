import logging
logger = logging.getLogger("uvicorn.error")
import datetime

import tokencost


class PricingManager:
    """
    Manages OpenAI pricing with fallback strategies.
    """

    def __init__(self):
        self.local_cache = OPENAI_PRICING ={}  # Your baseline
        self.last_updated = None
        self.cache_ttl_hours = 168  # 1 week

    def get_pricing(self, model: str) -> dict:
        """Get pricing with fallback to local cache."""

        # Try to fetch fresh data periodically
        if self._should_refresh():
            self._refresh_pricing()

        # Return from cache (always available)
        return self.local_cache.get(model)

    def _should_refresh(self) -> bool:
        if not self.last_updated:
            return True

        from datetime import datetime, timedelta
        return datetime.now() - self.last_updated > timedelta(hours=self.cache_ttl_hours)

    def _refresh_pricing(self):
        """Attempt to fetch updated pricing."""
        try:
            # Try llms.txt first
            fresh_pricing = self._fetch_llms_txt()
            if fresh_pricing:
                self.local_cache.update(fresh_pricing)
                self.last_updated = datetime.datetime.now()
        except Exception as e:
            # Log error but continue with cached data
            logger.debug(f"Pricing refresh failed: {e}")

    def _fetch_llms_txt(self) -> dict:
        """Parse llms.txt file."""
        # Implement parsing logic based on current format
        return tokencost.TOKEN_COSTS

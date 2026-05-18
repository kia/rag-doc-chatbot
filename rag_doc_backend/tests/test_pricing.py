from rag_doc_backend.estimation.pricing_manager import PricingManager


class TestPricingManager:
    def test_pricing_manager_initialization(self):
        manager = PricingManager()
        assert manager.local_cache is not None
        assert isinstance(manager.local_cache, dict)

    def test_gpt_3_5_turbo(self):
        manager = PricingManager()
        pricing = manager.get_pricing("gpt-3.5-turbo")
        assert pricing is not None
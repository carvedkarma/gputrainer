import unittest

from portfolio import PortfolioManager


class PortfolioCooldownTests(unittest.TestCase):
    def test_zero_cooldown_allows_next_bar_entry(self):
        pm = PortfolioManager(cooldown_bars=0)
        pm.set_bar(10)
        pm.last_trade_bar["BTCUSDT"] = 10
        self.assertTrue(pm.cooldown_ok("BTCUSDT"))

    def test_positive_cooldown_requires_elapsed_bars(self):
        pm = PortfolioManager(cooldown_bars=2)
        pm.set_bar(10)
        pm.last_trade_bar["ETHUSDT"] = 9
        self.assertFalse(pm.cooldown_ok("ETHUSDT"))
        pm.set_bar(11)
        self.assertTrue(pm.cooldown_ok("ETHUSDT"))


if __name__ == "__main__":
    unittest.main()

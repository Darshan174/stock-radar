"""
Algorithmic Stock Scorer.
Provides formula-based scoring for stock analysis - no LLM interpretation.
These scores are calculated using quantitative financial metrics.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScoringWeights:
    """Configurable weights for score calculation."""
    momentum: float = 0.35
    value: float = 0.25
    quality: float = 0.25
    risk: float = 0.15

    def __post_init__(self):
        total = self.momentum + self.value + self.quality + self.risk
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Scoring weights sum to {total}, normalizing to 1.0")
            self.momentum /= total
            self.value /= total
            self.quality /= total
            self.risk /= total


# Default weight presets
WEIGHT_PRESETS = {
    "balanced": ScoringWeights(momentum=0.35, value=0.25, quality=0.25, risk=0.15),
    "momentum_focus": ScoringWeights(momentum=0.50, value=0.15, quality=0.15, risk=0.20),
    "value_focus": ScoringWeights(momentum=0.20, value=0.40, quality=0.25, risk=0.15),
    "quality_focus": ScoringWeights(momentum=0.20, value=0.20, quality=0.45, risk=0.15),
    "conservative": ScoringWeights(momentum=0.25, value=0.25, quality=0.25, risk=0.25),
}


@dataclass
class AlgorithmicScores:
    """Formula-based scores for stock analysis."""
    momentum_score: int  # 0-100
    value_score: int  # 0-100
    quality_score: int  # 0-100
    risk_score: int  # 1-10
    confidence_score: int  # 0-100
    composite_score: float  # Weighted composite score
    overall_signal: str  # strong_buy, buy, hold, sell, strong_sell

    # Score breakdowns
    momentum_breakdown: Dict[str, Any]
    value_breakdown: Dict[str, Any]
    quality_breakdown: Dict[str, Any]
    risk_breakdown: Dict[str, Any]

    # Weights used for calculation
    weights_used: Dict[str, float] = field(default_factory=dict)


class StockScorer:
    """
    Algorithmic scorer using financial formulas.
    All scores are mathematically derived from actual data.
    """

    def __init__(self, weights: Optional[ScoringWeights] = None, preset: str = "balanced"):
        """
        Initialize scorer with configurable weights.

        Args:
            weights: Custom ScoringWeights instance
            preset: Use a preset ("balanced", "momentum_focus", "value_focus",
                   "quality_focus", "conservative")
        """
        if weights:
            self.weights = weights
        elif preset in WEIGHT_PRESETS:
            self.weights = WEIGHT_PRESETS[preset]
        else:
            self.weights = WEIGHT_PRESETS["balanced"]

        logger.info(f"StockScorer initialized with weights: M={self.weights.momentum:.0%}, "
                   f"V={self.weights.value:.0%}, Q={self.weights.quality:.0%}, R={self.weights.risk:.0%}")

    def calculate_momentum_score(self, indicators: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
        """
        Calculate momentum score based on technical indicators.
        
        Formula components:
        - RSI: Oversold (<30) = bullish, Overbought (>70) = bearish
        - MACD: Above signal = bullish, Below signal = bearish
        - Price vs SMA: Above SMA = bullish trend
        
        Returns:
            Tuple of (score 0-100, breakdown dict)
        """
        score = 50  # Base neutral score
        breakdown = {}
        
        # RSI Component (max 30 points)
        rsi = indicators.get('rsi_14')
        if rsi is not None:
            if rsi < 30:
                rsi_points = 25  # Strong oversold - bullish
                rsi_signal = "Oversold - Bullish reversal likely"
            elif rsi < 40:
                rsi_points = 15
                rsi_signal = "Near oversold - Watch for entry"
            elif rsi > 70:
                rsi_points = -20  # Overbought - bearish
                rsi_signal = "Overbought - Potential pullback"
            elif rsi > 60:
                rsi_points = 10
                rsi_signal = "Bullish momentum"
            else:
                rsi_points = 0
                rsi_signal = "Neutral"
            
            score += rsi_points
            breakdown['rsi'] = {
                'value': round(rsi, 1),
                'points': rsi_points,
                'signal': rsi_signal
            }
        
        # MACD Component (max 20 points)
        macd = indicators.get('macd')
        macd_signal = indicators.get('macd_signal')
        if macd is not None and macd_signal is not None:
            macd_diff = macd - macd_signal
            if macd_diff > 0:
                macd_points = min(15, int(macd_diff * 2))  # Cap at 15
                macd_status = "Bullish crossover"
            else:
                macd_points = max(-15, int(macd_diff * 2))
                macd_status = "Bearish crossover"
            
            score += macd_points
            breakdown['macd'] = {
                'value': round(macd, 2),
                'signal_line': round(macd_signal, 2),
                'points': macd_points,
                'status': macd_status
            }
        
        # Price vs SMA Component (max 20 points)
        price_vs_sma20 = indicators.get('price_vs_sma20_pct')
        if price_vs_sma20 is not None:
            if price_vs_sma20 > 5:
                sma_points = 15
                sma_signal = "Strong uptrend"
            elif price_vs_sma20 > 0:
                sma_points = int(price_vs_sma20 * 2)
                sma_signal = "Above SMA - Bullish"
            elif price_vs_sma20 < -5:
                sma_points = -15
                sma_signal = "Strong downtrend"
            else:
                sma_points = int(price_vs_sma20 * 2)
                sma_signal = "Below SMA - Bearish"
            
            score += sma_points
            breakdown['price_vs_sma'] = {
                'value': round(price_vs_sma20, 2),
                'points': sma_points,
                'signal': sma_signal
            }
        
        # SMA50 trend (max 10 points)
        price_vs_sma50 = indicators.get('price_vs_sma50_pct')
        if price_vs_sma50 is not None:
            sma50_points = min(10, max(-10, int(price_vs_sma50)))
            score += sma50_points
            breakdown['sma50_trend'] = {
                'value': round(price_vs_sma50, 2),
                'points': sma50_points
            }
        
        final_score = max(0, min(100, score))
        return final_score, breakdown
    
    def calculate_value_score(self, fundamentals: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
        """
        Calculate value score based on valuation metrics.
        
        Formula components:
        - P/E Ratio: Lower is better (compared to typical 20x)
        - P/B Ratio: Lower is better (under 1.5 = undervalued)
        - Dividend Yield: Higher is better
        - P/S Ratio: Lower is better
        
        Returns:
            Tuple of (score 0-100, breakdown dict)
        """
        score = 50
        breakdown = {}
        
        if not fundamentals:
            return 50, {'error': 'No fundamentals data'}
        
        # P/E Ratio (max 25 points)
        pe = fundamentals.get('pe_ratio')
        if pe is not None and pe > 0:
            if pe < 10:
                pe_points = 25
                pe_signal = "Very cheap valuation"
            elif pe < 15:
                pe_points = 20
                pe_signal = "Undervalued"
            elif pe < 20:
                pe_points = 10
                pe_signal = "Fair value"
            elif pe < 30:
                pe_points = 0
                pe_signal = "Slightly expensive"
            elif pe < 50:
                pe_points = -10
                pe_signal = "Expensive"
            else:
                pe_points = -20
                pe_signal = "Very expensive"
            
            score += pe_points
            breakdown['pe_ratio'] = {
                'value': round(pe, 2),
                'points': pe_points,
                'signal': pe_signal
            }
        
        # P/B Ratio (max 15 points)
        pb = fundamentals.get('pb_ratio')
        if pb is not None and pb > 0:
            if pb < 1:
                pb_points = 15
                pb_signal = "Trading below book value"
            elif pb < 1.5:
                pb_points = 10
                pb_signal = "Near book value"
            elif pb < 3:
                pb_points = 0
                pb_signal = "Fair P/B"
            else:
                pb_points = -10
                pb_signal = "High P/B premium"
            
            score += pb_points
            breakdown['pb_ratio'] = {
                'value': round(pb, 2),
                'points': pb_points,
                'signal': pb_signal
            }
        
        # Dividend Yield (max 15 points)
        div_yield = fundamentals.get('dividend_yield')
        if div_yield is not None:
            if div_yield > 0.05:
                div_points = 15
                div_signal = "High dividend yield"
            elif div_yield > 0.03:
                div_points = 10
                div_signal = "Good dividend"
            elif div_yield > 0.01:
                div_points = 5
                div_signal = "Modest dividend"
            else:
                div_points = 0
                div_signal = "Low/No dividend"
            
            score += div_points
            breakdown['dividend_yield'] = {
                'value': round(div_yield * 100, 2),
                'points': div_points,
                'signal': div_signal
            }
        
        # PEG Ratio (max 10 points) - P/E relative to growth
        peg = fundamentals.get('peg_ratio')
        if peg is not None and peg > 0:
            if peg < 1:
                peg_points = 10
                peg_signal = "Undervalued relative to growth"
            elif peg < 1.5:
                peg_points = 5
                peg_signal = "Fair PEG"
            else:
                peg_points = -5
                peg_signal = "Overvalued relative to growth"
            
            score += peg_points
            breakdown['peg_ratio'] = {
                'value': round(peg, 2),
                'points': peg_points,
                'signal': peg_signal
            }
        
        final_score = max(0, min(100, score))
        return final_score, breakdown
    
    def calculate_quality_score(self, fundamentals: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
        """
        Calculate quality score based on profitability and financial health.
        
        Formula components:
        - ROE: Higher is better (>15% = good, >20% = excellent)
        - Profit Margin: Higher is better
        - Debt/Equity: Lower is better
        - Current Ratio: Above 1.5 = healthy
        
        Returns:
            Tuple of (score 0-100, breakdown dict)
        """
        score = 50
        breakdown = {}
        
        if not fundamentals:
            return 50, {'error': 'No fundamentals data'}
        
        # ROE - Return on Equity (max 25 points)
        roe = fundamentals.get('roe')
        if roe is not None:
            if roe > 0.25:
                roe_points = 25
                roe_signal = "Excellent returns"
            elif roe > 0.20:
                roe_points = 20
                roe_signal = "Very good ROE"
            elif roe > 0.15:
                roe_points = 15
                roe_signal = "Good ROE"
            elif roe > 0.10:
                roe_points = 5
                roe_signal = "Adequate ROE"
            elif roe > 0:
                roe_points = -5
                roe_signal = "Low ROE"
            else:
                roe_points = -15
                roe_signal = "Negative ROE"
            
            score += roe_points
            breakdown['roe'] = {
                'value': round(roe * 100, 2),
                'points': roe_points,
                'signal': roe_signal
            }
        
        # Profit Margin (max 20 points)
        margin = fundamentals.get('profit_margin')
        if margin is not None:
            if margin > 0.25:
                margin_points = 20
                margin_signal = "Excellent margins"
            elif margin > 0.15:
                margin_points = 15
                margin_signal = "Strong margins"
            elif margin > 0.10:
                margin_points = 10
                margin_signal = "Good margins"
            elif margin > 0:
                margin_points = 0
                margin_signal = "Thin margins"
            else:
                margin_points = -20
                margin_signal = "Negative margins"
            
            score += margin_points
            breakdown['profit_margin'] = {
                'value': round(margin * 100, 2),
                'points': margin_points,
                'signal': margin_signal
            }
        
        # Debt/Equity Ratio (max 15 points)
        de = fundamentals.get('debt_to_equity')
        if de is not None:
            if de < 0.3:
                de_points = 15
                de_signal = "Low debt - Very safe"
            elif de < 0.5:
                de_points = 10
                de_signal = "Conservative debt"
            elif de < 1:
                de_points = 5
                de_signal = "Moderate debt"
            elif de < 2:
                de_points = -5
                de_signal = "High debt"
            else:
                de_points = -15
                de_signal = "Very high debt - Risky"
            
            score += de_points
            breakdown['debt_to_equity'] = {
                'value': round(de, 2),
                'points': de_points,
                'signal': de_signal
            }
        
        # Current Ratio (max 10 points) - Liquidity
        current = fundamentals.get('current_ratio')
        if current is not None:
            if current > 2:
                current_points = 10
                current_signal = "Strong liquidity"
            elif current > 1.5:
                current_points = 5
                current_signal = "Good liquidity"
            elif current > 1:
                current_points = 0
                current_signal = "Adequate liquidity"
            else:
                current_points = -10
                current_signal = "Liquidity risk"
            
            score += current_points
            breakdown['current_ratio'] = {
                'value': round(current, 2),
                'points': current_points,
                'signal': current_signal
            }
        
        final_score = max(0, min(100, score))
        return final_score, breakdown
    
    def calculate_risk_score(
        self,
        quote: Dict[str, Any],
        indicators: Dict[str, Any],
        fundamentals: Dict[str, Any]
    ) -> tuple[int, Dict[str, Any]]:
        """
        Calculate risk score (1 = low risk, 10 = high risk).

        Formula components:
        - ATR-based volatility (primary volatility measure)
        - ADX trend strength (weak trends = higher risk)
        - Debt levels
        - Profit margin (negative = risky)
        - Valuation (high P/E = risky)
        - Volume/liquidity risk

        Returns:
            Tuple of (risk 1-10, breakdown dict)
        """
        risk = 5  # Base moderate risk
        breakdown = {}

        # =================================================================
        # VOLATILITY RISK - Using ATR (Primary measure)
        # =================================================================
        atr_pct = indicators.get('atr_pct')
        if atr_pct is not None:
            # ATR% thresholds (daily volatility as % of price)
            # < 1% = very low volatility
            # 1-2% = normal volatility
            # 2-4% = elevated volatility
            # > 4% = high volatility
            if atr_pct > 5:
                risk += 2
                breakdown['volatility_atr'] = {
                    'value': round(atr_pct, 2),
                    'factor': 'Very high volatility (ATR > 5%)',
                    'risk_add': 2
                }
            elif atr_pct > 3:
                risk += 1.5
                breakdown['volatility_atr'] = {
                    'value': round(atr_pct, 2),
                    'factor': 'High volatility (ATR 3-5%)',
                    'risk_add': 1.5
                }
            elif atr_pct > 2:
                risk += 0.5
                breakdown['volatility_atr'] = {
                    'value': round(atr_pct, 2),
                    'factor': 'Elevated volatility (ATR 2-3%)',
                    'risk_add': 0.5
                }
            elif atr_pct < 1:
                risk -= 0.5
                breakdown['volatility_atr'] = {
                    'value': round(atr_pct, 2),
                    'factor': 'Low volatility (ATR < 1%)',
                    'risk_add': -0.5
                }
        else:
            # Fallback to RSI-based volatility if ATR not available
            rsi = indicators.get('rsi_14', 50)
            if rsi < 25 or rsi > 75:
                risk += 1
                breakdown['volatility_rsi'] = {
                    'value': round(rsi, 1),
                    'factor': 'High volatility (extreme RSI)',
                    'risk_add': 1
                }
            elif rsi < 35 or rsi > 65:
                risk += 0.5
                breakdown['volatility_rsi'] = {
                    'value': round(rsi, 1),
                    'factor': 'Moderate volatility',
                    'risk_add': 0.5
                }

        # =================================================================
        # TREND RISK - Using ADX (Weak trends = unpredictable = higher risk)
        # =================================================================
        adx = indicators.get('adx')
        if adx is not None:
            # ADX thresholds:
            # < 20 = no/weak trend (higher risk - unpredictable)
            # 20-25 = emerging trend
            # 25-50 = strong trend (lower risk - predictable)
            # > 50 = very strong trend (potential exhaustion risk)
            if adx < 20:
                risk += 1
                breakdown['trend_strength'] = {
                    'value': round(adx, 1),
                    'factor': 'Weak/No trend (ADX < 20) - Unpredictable',
                    'risk_add': 1
                }
            elif adx > 50:
                risk += 0.5
                breakdown['trend_strength'] = {
                    'value': round(adx, 1),
                    'factor': 'Very strong trend (ADX > 50) - Potential exhaustion',
                    'risk_add': 0.5
                }
            elif adx >= 25:
                risk -= 0.5
                breakdown['trend_strength'] = {
                    'value': round(adx, 1),
                    'factor': 'Strong trend (ADX 25-50) - Predictable',
                    'risk_add': -0.5
                }

        # =================================================================
        # DIRECTIONAL RISK - Using +DI/-DI divergence
        # =================================================================
        plus_di = indicators.get('plus_di')
        minus_di = indicators.get('minus_di')
        if plus_di is not None and minus_di is not None:
            di_diff = abs(plus_di - minus_di)
            if di_diff < 5:
                # Very close +DI and -DI = indecision = higher risk
                risk += 0.5
                breakdown['directional'] = {
                    'plus_di': round(plus_di, 1),
                    'minus_di': round(minus_di, 1),
                    'factor': 'Directional indecision (+DI ≈ -DI)',
                    'risk_add': 0.5
                }

        # =================================================================
        # DEBT RISK
        # =================================================================
        if fundamentals:
            de = fundamentals.get('debt_to_equity')
            if de is not None:
                if de > 2:
                    risk += 2
                    breakdown['debt'] = {'value': de, 'factor': 'Very high debt', 'risk_add': 2}
                elif de > 1:
                    risk += 1
                    breakdown['debt'] = {'value': de, 'factor': 'High debt', 'risk_add': 1}
                elif de < 0.3:
                    risk -= 0.5
                    breakdown['debt'] = {'value': de, 'factor': 'Very low debt', 'risk_add': -0.5}

        # =================================================================
        # PROFITABILITY RISK
        # =================================================================
        if fundamentals:
            margin = fundamentals.get('profit_margin')
            if margin is not None:
                if margin < 0:
                    risk += 2
                    breakdown['profitability'] = {
                        'value': round(margin * 100, 1),
                        'factor': 'Negative profit margin',
                        'risk_add': 2
                    }
                elif margin < 0.05:
                    risk += 0.5
                    breakdown['profitability'] = {
                        'value': round(margin * 100, 1),
                        'factor': 'Very thin margins (< 5%)',
                        'risk_add': 0.5
                    }

        # =================================================================
        # VALUATION RISK
        # =================================================================
        if fundamentals:
            pe = fundamentals.get('pe_ratio')
            if pe is not None:
                if pe > 100:
                    risk += 2
                    breakdown['valuation'] = {'value': pe, 'factor': 'Extremely high P/E', 'risk_add': 2}
                elif pe > 50:
                    risk += 1
                    breakdown['valuation'] = {'value': pe, 'factor': 'High P/E', 'risk_add': 1}
                elif pe < 0:
                    risk += 1.5
                    breakdown['valuation'] = {'value': pe, 'factor': 'Negative P/E (losses)', 'risk_add': 1.5}

        # =================================================================
        # LIQUIDITY RISK
        # =================================================================
        if quote:
            volume = quote.get('volume', 0)
            avg_volume = quote.get('avg_volume', 1)
            if avg_volume > 0:
                vol_ratio = volume / avg_volume
                if vol_ratio < 0.3:
                    risk += 1
                    breakdown['liquidity'] = {
                        'volume_ratio': round(vol_ratio, 2),
                        'factor': 'Very low volume (< 30% avg)',
                        'risk_add': 1
                    }
                elif vol_ratio < 0.5:
                    risk += 0.5
                    breakdown['liquidity'] = {
                        'volume_ratio': round(vol_ratio, 2),
                        'factor': 'Low volume (< 50% avg)',
                        'risk_add': 0.5
                    }

        final_risk = max(1, min(10, int(round(risk))))
        return final_risk, breakdown
    
    def calculate_confidence_score(
        self,
        has_fundamentals: bool,
        has_indicators: bool,
        price_history_days: int,
        has_news: bool
    ) -> tuple[int, Dict[str, Any]]:
        """
        Calculate confidence score based on data quality/availability.
        
        More data = higher confidence in analysis.
        
        Returns:
            Tuple of (confidence 0-100, breakdown dict)
        """
        score = 30  # Base confidence
        breakdown = {}
        
        # Fundamentals data (+25)
        if has_fundamentals:
            score += 25
            breakdown['fundamentals'] = {'available': True, 'points': 25}
        else:
            breakdown['fundamentals'] = {'available': False, 'points': 0}
        
        # Technical indicators (+20)
        if has_indicators:
            score += 20
            breakdown['indicators'] = {'available': True, 'points': 20}
        else:
            breakdown['indicators'] = {'available': False, 'points': 0}
        
        # Price history depth (+20)
        if price_history_days >= 365:
            history_points = 20
            history_note = "1+ year history"
        elif price_history_days >= 180:
            history_points = 15
            history_note = "6+ months history"
        elif price_history_days >= 90:
            history_points = 10
            history_note = "3+ months history"
        elif price_history_days >= 30:
            history_points = 5
            history_note = "1+ month history"
        else:
            history_points = 0
            history_note = "Limited history"
        
        score += history_points
        breakdown['history'] = {
            'days': price_history_days,
            'points': history_points,
            'note': history_note
        }
        
        # News data (+10)
        if has_news:
            score += 10
            breakdown['news'] = {'available': True, 'points': 10}
        else:
            breakdown['news'] = {'available': False, 'points': 0}
        
        final_score = max(0, min(100, score))
        return final_score, breakdown
    
    def determine_signal(
        self,
        momentum: int,
        value: int,
        quality: int,
        risk: int
    ) -> tuple[str, float]:
        """
        Determine overall trading signal from scores using configurable weights.

        Returns:
            Tuple of (signal string, composite score)
        """
        # Weighted composite score using configurable weights
        composite = (
            momentum * self.weights.momentum +
            value * self.weights.value +
            quality * self.weights.quality +
            (100 - (risk * 10)) * self.weights.risk  # Invert risk: low risk = high score
        )

        if composite >= 75:
            signal = "strong_buy"
        elif composite >= 60:
            signal = "buy"
        elif composite >= 40:
            signal = "hold"
        elif composite >= 25:
            signal = "sell"
        else:
            signal = "strong_sell"

        return signal, round(composite, 2)
    
    def calculate_all_scores(
        self,
        quote: Dict[str, Any],
        indicators: Dict[str, Any],
        fundamentals: Optional[Dict[str, Any]] = None,
        price_history_days: int = 0,
        has_news: bool = False
    ) -> AlgorithmicScores:
        """
        Calculate all algorithmic scores for a stock.

        Returns:
            AlgorithmicScores dataclass with all metrics
        """
        momentum, momentum_bd = self.calculate_momentum_score(indicators)
        value, value_bd = self.calculate_value_score(fundamentals or {})
        quality, quality_bd = self.calculate_quality_score(fundamentals or {})
        risk, risk_bd = self.calculate_risk_score(quote, indicators, fundamentals or {})
        confidence, conf_bd = self.calculate_confidence_score(
            has_fundamentals=bool(fundamentals),
            has_indicators=bool(indicators),
            price_history_days=price_history_days,
            has_news=has_news
        )

        signal, composite = self.determine_signal(momentum, value, quality, risk)

        return AlgorithmicScores(
            momentum_score=momentum,
            value_score=value,
            quality_score=quality,
            risk_score=risk,
            confidence_score=confidence,
            composite_score=composite,
            overall_signal=signal,
            momentum_breakdown=momentum_bd,
            value_breakdown=value_bd,
            quality_breakdown=quality_bd,
            risk_breakdown=risk_bd,
            weights_used={
                'momentum': self.weights.momentum,
                'value': self.weights.value,
                'quality': self.weights.quality,
                'risk': self.weights.risk
            }
        )


# =============================================================================
# Quick Test
# =============================================================================

if __name__ == "__main__":
    # Test with different weight presets
    print("\n" + "=" * 60)
    print("TESTING SCORING WITH DIFFERENT WEIGHT PRESETS")
    print("=" * 60)

    # Test data with new indicators (ATR, ADX)
    test_indicators = {
        'rsi_14': 55,
        'macd': 2.5,
        'macd_signal': 1.8,
        'price_vs_sma20_pct': 3.2,
        'price_vs_sma50_pct': 5.1,
        'atr_14': 45.5,
        'atr_pct': 1.8,  # 1.8% daily volatility
        'adx': 32,  # Strong trend
        'plus_di': 28,
        'minus_di': 18,
    }

    test_fundamentals = {
        'pe_ratio': 18.5,
        'pb_ratio': 1.8,
        'dividend_yield': 0.025,
        'roe': 0.18,
        'profit_margin': 0.12,
        'debt_to_equity': 0.45,
        'current_ratio': 1.6,
    }

    test_quote = {
        'price': 2500,
        'volume': 1500000,
        'avg_volume': 1200000,
    }

    # Test each preset
    for preset_name in ["balanced", "momentum_focus", "value_focus", "conservative"]:
        scorer = StockScorer(preset=preset_name)

        scores = scorer.calculate_all_scores(
            quote=test_quote,
            indicators=test_indicators,
            fundamentals=test_fundamentals,
            price_history_days=365,
            has_news=True
        )

        print(f"\n--- {preset_name.upper()} PRESET ---")
        print(f"Weights: M={scores.weights_used['momentum']:.0%}, "
              f"V={scores.weights_used['value']:.0%}, "
              f"Q={scores.weights_used['quality']:.0%}, "
              f"R={scores.weights_used['risk']:.0%}")
        print(f"Scores: M={scores.momentum_score}, V={scores.value_score}, "
              f"Q={scores.quality_score}, Risk={scores.risk_score}/10")
        print(f"Composite: {scores.composite_score:.1f} -> {scores.overall_signal.upper()}")

    # Show risk breakdown with new volatility metrics
    print("\n" + "=" * 60)
    print("RISK BREAKDOWN (with ATR/ADX volatility)")
    print("=" * 60)
    scorer = StockScorer()
    _, risk_bd = scorer.calculate_risk_score(test_quote, test_indicators, test_fundamentals)
    for key, value in risk_bd.items():
        print(f"  {key}: {value}")

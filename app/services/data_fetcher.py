import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import akshare as ak
import pandas as pd

from app import config

logger = logging.getLogger(__name__)


class DataFetcher:
    """Wraps akshare to provide stock data for Kronos Stock Advisor."""

    def __init__(self):
        self.cache_dir = config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

    # ── Cache helpers ──────────────────────────────────────────────

    def _cache_path(self, key: str) -> str:
        safe_key = key.replace("/", "_").replace("\\", "_")
        return os.path.join(self.cache_dir, f"{safe_key}.json")

    def _get_cache(self, key: str, max_age_hours: float) -> Optional[pd.DataFrame | dict | list]:
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            mtime = os.path.getmtime(path)
            if time.time() - mtime > max_age_hours * 3600:
                return None
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            fmt = payload.get("_fmt")
            data = payload.get("data")
            if fmt == "dataframe":
                return pd.DataFrame(data)
            return data
        except Exception as e:
            logger.warning("Cache read failed for %s: %s", key, e)
            return None

    def _set_cache(self, key: str, data) -> None:
        path = self._cache_path(key)
        try:
            if isinstance(data, pd.DataFrame):
                payload = {"_fmt": "dataframe", "data": data.to_dict(orient="records")}
            else:
                payload = {"_fmt": "raw", "data": data}
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, default=str)
        except Exception as e:
            logger.warning("Cache write failed for %s: %s", key, e)

    # ── Market helper ──────────────────────────────────────────────

    @staticmethod
    def _get_market(symbol: str) -> str:
        """Return 'sh' or 'sz' based on A-share stock code prefix."""
        if symbol.startswith("6"):
            return "sh"
        return "sz"

    @staticmethod
    def _is_hk(symbol: str) -> bool:
        """Check if a symbol is a HK stock (5-digit code like 01810, 00700)."""
        return len(symbol) == 5 and symbol.isdigit()

    # ── Public API ─────────────────────────────────────────────────

    def get_stock_history(
        self, symbol: str, lookback: int = 400
    ) -> Optional[pd.DataFrame]:
        """Get daily OHLCV history for an A-share stock.

        Tries eastmoney (stock_zh_a_hist) first, falls back to sina (stock_zh_a_daily).

        Returns DataFrame with columns:
            [date, open, high, low, close, volume, amount]
        """
        cache_key = f"hist_{symbol}_{lookback}"
        cached = self._get_cache(cache_key, max_age_hours=24)
        if cached is not None:
            return cached

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=int(lookback * 1.8))).strftime("%Y%m%d")

        if self._is_hk(symbol):
            df = self._fetch_hist_hk(symbol)
        else:
            df = self._fetch_hist_eastmoney(symbol, start_date, end_date)
            if df is None or df.empty:
                logger.info("Eastmoney failed for %s, trying sina fallback", symbol)
                df = self._fetch_hist_sina(symbol, start_date, end_date)

        if df is None or df.empty:
            logger.warning("No history data for %s from any source", symbol)
            return None

        df = df.tail(lookback).reset_index(drop=True)
        self._set_cache(cache_key, df)
        return df

    def _fetch_hist_eastmoney(
        self, symbol: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Fetch from eastmoney via stock_zh_a_hist."""
        df = None
        for attempt in range(3):
            try:
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="",
                )
                if df is not None and not df.empty:
                    break
            except Exception as e:
                logger.warning("Eastmoney attempt %d failed for %s: %s", attempt + 1, symbol, e)
                time.sleep(1.5 * (attempt + 1))

        if df is None or df.empty:
            return None

        col_map = {
            "日期": "date", "开盘": "open", "最高": "high",
            "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount",
        }
        df = df.rename(columns=col_map)
        keep = [c for c in ["date", "open", "high", "low", "close", "volume", "amount"] if c in df.columns]
        return df[keep]

    def _fetch_hist_sina(
        self, symbol: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Fetch from sina via stock_zh_a_daily (fallback)."""
        market = self._get_market(symbol)
        sina_symbol = f"{'sh' if market == 'sh' else 'sz'}{symbol}"
        try:
            df = ak.stock_zh_a_daily(
                symbol=sina_symbol,
                start_date=start_date,
                end_date=end_date,
                adjust="",
            )
            if df is None or df.empty:
                return None

            col_map = {}
            for col in df.columns:
                cl = str(col).lower()
                if cl == "date":
                    col_map[col] = "date"
                elif cl == "open":
                    col_map[col] = "open"
                elif cl == "high":
                    col_map[col] = "high"
                elif cl == "low":
                    col_map[col] = "low"
                elif cl == "close":
                    col_map[col] = "close"
                elif cl == "volume":
                    col_map[col] = "volume"
                elif cl == "amount":
                    col_map[col] = "amount"

            df = df.rename(columns=col_map)
            keep = [c for c in ["date", "open", "high", "low", "close", "volume", "amount"] if c in df.columns]
            return df[keep]
        except Exception as e:
            logger.warning("Sina fallback failed for %s: %s", symbol, e)
            return None

    def _fetch_hist_hk(
        self, symbol: str
    ) -> Optional[pd.DataFrame]:
        """Fetch HK stock history from sina via stock_hk_daily."""
        try:
            df = ak.stock_hk_daily(symbol=symbol, adjust="")
            if df is None or df.empty:
                return None
            # Columns are already: date, open, high, low, close, volume, amount
            keep = [c for c in ["date", "open", "high", "low", "close", "volume", "amount"] if c in df.columns]
            return df[keep]
        except Exception as e:
            logger.warning("HK hist fetch failed for %s: %s", symbol, e)
            return None

    def get_stock_realtime(self, symbol: str) -> Optional[Dict]:
        """Get latest quote for a stock (from recent history as fallback)."""
        try:
            df = self.get_stock_history(symbol, lookback=5)
            if df is None or df.empty:
                return None

            row = df.iloc[-1]
            prev_close = float(df.iloc[-2]["close"]) if len(df) >= 2 else float(row["close"])
            price = float(row["close"])
            change_pct = (price / prev_close - 1) * 100 if prev_close > 0 else 0

            return {
                "price": price,
                "open": float(row.get("open", 0)),
                "high": float(row.get("high", 0)),
                "low": float(row.get("low", 0)),
                "volume": float(row.get("volume", 0)),
                "change_pct": round(change_pct, 2),
            }
        except Exception as e:
            logger.error("Failed to fetch realtime for %s: %s", symbol, e)
            return None

    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """Get basic stock info (name, industry).

        Tries eastmoney individual_info first, falls back to code-name list.
        """
        cache_key = f"info_{symbol}"
        cached = self._get_cache(cache_key, max_age_hours=24)
        if cached is not None:
            return cached

        info = {}

        # Primary: eastmoney detailed info
        try:
            df = ak.stock_individual_info_em(symbol=symbol)
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    key = str(row.iloc[0])
                    val = str(row.iloc[1])
                    if "名称" in key or "股票简称" in key:
                        info["name"] = val
                    elif "行业" in key:
                        info["industry"] = val
                    elif "上市日期" in key or "上市时间" in key:
                        info["list_date"] = val
                    elif "总市值" in key:
                        info["market_cap"] = val
        except Exception as e:
            logger.warning("Eastmoney info failed for %s: %s", symbol, e)

        # Fallback: get name from code-name list (A-share) or HK list
        if not info.get("name"):
            if self._is_hk(symbol):
                name = self._lookup_name_from_hk_list(symbol)
            else:
                name = self._lookup_name_from_list(symbol)
            if name:
                info["name"] = name

        info.setdefault("name", "")
        info.setdefault("industry", "")

        if info.get("name"):
            self._set_cache(cache_key, info)

        return info

    def _lookup_name_from_list(self, symbol: str) -> Optional[str]:
        """Lookup stock name from the cached code-name list."""
        try:
            code_name_list = self._get_a_share_list()
            if code_name_list:
                for item in code_name_list:
                    if str(item.get("code", "")) == symbol:
                        return str(item.get("name", ""))
        except Exception as e:
            logger.warning("Code-name lookup failed for %s: %s", symbol, e)
        return None

    def _lookup_name_from_hk_list(self, symbol: str) -> Optional[str]:
        """Lookup HK stock name from cached HK list."""
        try:
            hk_list = self._get_cache("hk_code_name_list", max_age_hours=24)
            if hk_list:
                for item in hk_list:
                    if str(item.get("code", "")) == symbol:
                        return str(item.get("name", ""))
        except Exception as e:
            logger.warning("HK name lookup failed for %s: %s", symbol, e)
        return None

    def search_stock(self, keyword: str) -> List[Dict]:
        """Search stocks by code or name (A-share + HK via港股通)."""
        results = []

        # A-share search
        a_list = self._get_a_share_list()
        for item in a_list:
            code = str(item.get("code", ""))
            name = str(item.get("name", ""))
            if keyword in code or keyword in name:
                results.append({"symbol": code, "name": name, "price": "", "market": "a_share"})
            if len(results) >= 15:
                break

        # HK search — only if HK list is already cached (avoid blocking)
        hk_list = self._get_cache("hk_code_name_list", max_age_hours=24)
        if hk_list:
            for item in hk_list:
                code = str(item.get("code", ""))
                name = str(item.get("name", ""))
                if keyword in code or keyword in name:
                    results.append({"symbol": code, "name": name, "price": "", "market": "hk"})
                if len(results) >= 20:
                    break

        return results

    def preload_hk_list(self):
        """Pre-load HK stock list in background (call once at startup)."""
        self._get_hk_list()

    def _get_a_share_list(self) -> list:
        """Get A-share code-name list (cached)."""
        cache_key = "stock_code_name_list"
        cached = self._get_cache(cache_key, max_age_hours=24)
        if cached is not None:
            return cached
        try:
            df = ak.stock_info_a_code_name()
            if df is None or df.empty:
                return []
            code_name_list = df.to_dict(orient="records")
            self._set_cache(cache_key, code_name_list)
            return code_name_list
        except Exception as e:
            logger.error("Failed to fetch A-share code-name list: %s", e)
            return []

    def _get_hk_list(self) -> list:
        """Get HK stock code-name list via sina (cached)."""
        cache_key = "hk_code_name_list"
        cached = self._get_cache(cache_key, max_age_hours=24)
        if cached is not None:
            return cached
        try:
            df = ak.stock_hk_spot()
            if df is None or df.empty:
                return []
            hk_list = []
            code_col = "代码" if "代码" in df.columns else df.columns[1]
            name_col = "中文名称" if "中文名称" in df.columns else df.columns[2]
            for _, row in df.iterrows():
                hk_list.append({"code": str(row[code_col]), "name": str(row[name_col])})
            self._set_cache(cache_key, hk_list)
            return hk_list
        except Exception as e:
            logger.warning("Failed to fetch HK stock list: %s", e)
            return []

    def get_index_components(self, index: str = "hs300") -> List[str]:
        """Get constituent stock codes of a major index."""
        cache_key = f"index_{index}"
        cached = self._get_cache(cache_key, max_age_hours=24)
        if cached is not None:
            return cached

        index_map = {
            "hs300": "000300",
            "zz500": "000905",
        }
        code = index_map.get(index)
        if code is None:
            logger.warning("Unknown index: %s", index)
            return []

        try:
            df = ak.index_stock_cons_csindex(symbol=code)
            if df is None or df.empty:
                return []

            col = None
            for c in df.columns:
                if "成分券代码" in str(c) or "代码" in str(c) or "constituent" in str(c).lower():
                    col = c
                    break
            if col is None:
                col = df.columns[0]

            components = df[col].astype(str).tolist()
            # Ensure 6-digit codes
            components = [c.zfill(6) for c in components]
            self._set_cache(cache_key, components)
            return components
        except Exception as e:
            logger.error("Failed to fetch index components for %s: %s", index, e)
            return []

    def get_fund_flow(self, symbol: str, days: int = 5) -> Optional[pd.DataFrame]:
        """Get recent fund flow (main force net inflow) data."""
        cache_key = f"fund_flow_{symbol}_{days}"
        cached = self._get_cache(cache_key, max_age_hours=2)
        if cached is not None:
            return cached

        if self._is_hk(symbol):
            return None

        market = self._get_market(symbol)
        try:
            df = ak.stock_individual_fund_flow(stock=symbol, market=market)
            if df is None or df.empty:
                return None

            # Normalize column names
            col_map = {}
            for c in df.columns:
                cs = str(c)
                if "主力净流入" in cs and "净额" in cs:
                    col_map[c] = "main_net_inflow"
                elif "日期" in cs:
                    col_map[c] = "date"
            df = df.rename(columns=col_map)

            df = df.tail(days).reset_index(drop=True)
            self._set_cache(cache_key, df)
            return df
        except Exception as e:
            logger.error("Failed to fetch fund flow for %s: %s", symbol, e)
            return None

    def get_margin_data(self, symbol: str, days: int = 20) -> Optional[pd.DataFrame]:
        """Get margin trading (融资融券) data.

        Tries stock_margin_detail_sse for SH, stock_margin_detail_szse for SZ.
        These APIs query by date, so we fetch recent dates and filter by symbol.
        """
        cache_key = f"margin_{symbol}_{days}"
        cached = self._get_cache(cache_key, max_age_hours=2)
        if cached is not None:
            return cached

        if self._is_hk(symbol):
            return None

        market = self._get_market(symbol)
        try:
            from datetime import date as dt_date
            import pandas as pd

            rows = []
            today = datetime.now()
            # Try last N business days to collect enough data
            for offset in range(days * 2):
                d = today - timedelta(days=offset)
                if d.weekday() >= 5:
                    continue
                date_str = d.strftime("%Y%m%d")
                try:
                    if market == "sh":
                        day_df = ak.stock_margin_detail_sse(date=date_str)
                    else:
                        day_df = ak.stock_margin_detail_szse(date=date_str)
                    if day_df is not None and not day_df.empty:
                        # Filter for our symbol
                        code_col = None
                        for c in day_df.columns:
                            if "代码" in str(c) or "标的" in str(c):
                                code_col = c
                                break
                        if code_col is None:
                            code_col = day_df.columns[0]
                        matched = day_df[day_df[code_col].astype(str).str.contains(symbol)]
                        if not matched.empty:
                            row = matched.iloc[0].to_dict()
                            row["_date"] = date_str
                            rows.append(row)
                except Exception:
                    continue
                if len(rows) >= days:
                    break

            if not rows:
                return None

            df = pd.DataFrame(rows)
            # Normalize: find margin balance column
            col_map = {"_date": "date"}
            for c in df.columns:
                cs = str(c)
                if "融资余额" in cs and "融券" not in cs:
                    col_map[c] = "margin_balance"
            df = df.rename(columns=col_map)

            self._set_cache(cache_key, df)
            return df
        except Exception as e:
            logger.warning("Failed to fetch margin data for %s: %s", symbol, e)
            return None

    def get_stock_sector(self, symbol: str) -> Optional[Dict]:
        """Get stock's industry sector and concept tags."""
        cache_key = f"sector_{symbol}"
        cached = self._get_cache(cache_key, max_age_hours=24)
        if cached is not None:
            return cached

        result: Dict = {"industry": "", "concepts": []}

        # Industry from stock info
        info = self.get_stock_info(symbol)
        if info:
            result["industry"] = info.get("industry", "")

        # Concept/板块 tags
        try:
            df = ak.stock_board_concept_name_em()
            if df is not None and not df.empty:
                concepts = []
                for _, row in df.iterrows():
                    board_name = str(row.get("板块名称", ""))
                    try:
                        cons = ak.stock_board_concept_cons_em(symbol=board_name)
                        if cons is not None and not cons.empty:
                            codes = cons["代码"].astype(str).tolist() if "代码" in cons.columns else []
                            if symbol in codes:
                                concepts.append(board_name)
                    except Exception:
                        continue
                    # Limit checks to avoid excessive API calls
                    if len(concepts) >= 5:
                        break
                result["concepts"] = concepts
        except Exception as e:
            logger.debug("Concept board lookup skipped for %s: %s", symbol, e)

        self._set_cache(cache_key, result)
        return result

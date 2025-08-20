'use client';

import React, { useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  BarChart,
  Bar,
  ComposedChart,
  Area,
} from "recharts";

/**
 * Dual Cryptocurrency — Pro Risk, Correlation & Volatility Workbench
 * -----------------------------------------------------------------
 * Data: CoinGecko (gratuit, sans clé) via /market_chart, /ohlc, /coins/markets
 * Pair analytics (A vs B):
 *   - Rendements log, corrélation, bêta (OLS), R², vol ann., Sharpe
 *   - Max drawdown, VaR & ES (CVaR) 95% 1j
 *   - Corrélation roulante (fenêtre réglable)
 *   - Régression (droite de marché) sur nuage retA vs retB + R²
 *   - Frontière efficiente 2-actifs + poids min-variance / max-Sharpe
 * Marché temps quasi-réel (poll 60s):
 *   - Ticker & table des prix, variation 1h/24h/7j, volume, market cap
 * Graphiques:
 *   - Chandeliers OHLC par actif (lightweight-charts) avec fallback line si indispo
 *   - Indice de prix normalisé, Vol 30j, Nuage ret/ret, Drawdowns
 *
 * Notes de robustesse:
 *   - Corrige l'erreur SyntaxError « Unterminated string constant » (CSV join avec "\n").
 *   - Corrige l'erreur TypeError « addCandlestickSeries n'est pas une fonction » via import dynamique + fallback.
 *   - Gestion réseau renforcée: retries exponentiels + messages d'erreur clairs.
 *   - Diagnostics intégrés (tests) pour valider les utilitaires stats.
 */

// ————————————————————————————————————————————————————————————————————
// Config
const COINS = [
  { id: "bitcoin", symbol: "BTC" },
  { id: "ethereum", symbol: "ETH" },
  { id: "solana", symbol: "SOL" },
  { id: "ripple", symbol: "XRP" },
  { id: "binancecoin", symbol: "BNB" },
  { id: "cardano", symbol: "ADA" },
  { id: "dogecoin", symbol: "DOGE" },
  { id: "polkadot", symbol: "DOT" },
  { id: "litecoin", symbol: "LTC" },
  { id: "chainlink", symbol: "LINK" },
  { id: "avalanche-2", symbol: "AVAX" },
  { id: "tron", symbol: "TRX" },
];

const DAY_OPTIONS = [
  { label: "30D", value: 30 },
  { label: "90D", value: 90 },
  { label: "180D", value: 180 },
  { label: "365D", value: 365 },
];

const REFRESH_MS_HIST = 10 * 60 * 1000; // 10 min
const REFRESH_MS_TICKS = 60 * 1000; // 1 min
const TRADING_DAYS = 252;

// ————————————————————————————————————————————————————————————————————
// Utils
function fmtPct(x, digits = 2) { if (x === null || Number.isNaN(x)) return "—"; return `${(x * 100).toFixed(digits)}%`; }
function fmtPctAbs(x, digits = 0) { if (x === null || Number.isNaN(x)) return "—"; return `${Math.abs(x).toFixed(digits)}%`; }
function fmtNum(x, digits = 2) {
  if (x === null || Number.isNaN(x)) return "—";
  const abs = Math.abs(x);
  const options = abs >= 1000 ? { maximumFractionDigits: 0 } : { maximumFractionDigits: digits };
  return x.toLocaleString(undefined, options);
}
function toISODate(ts) { return new Date(ts).toISOString().slice(0, 10); }
function toISODateTime(ts) { const d = new Date(ts * 1000); return d.toISOString().slice(0, 10); }
function seriesToDailyMap(prices) { // [ [ts, price], ... ] -> Map(date, price)
  const map = new Map();
  for (const [ts, p] of prices) map.set(toISODate(ts), p);
  return map;
}
function intersectSorted(aKeys, bKeys) {
  const a = [...aKeys].sort();
  const b = [...bKeys].sort();
  const out = [];
  let i = 0, j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] === b[j]) { out.push(a[i]); i++; j++; }
    else if (a[i] < b[j]) i++; else j++;
  }
  return out;
}
function computeLogReturns(prices) { const r = []; for (let i = 1; i < prices.length; i++) r.push(Math.log(prices[i] / prices[i - 1])); return r; }
function mean(arr) { return !arr.length ? NaN : arr.reduce((a, b) => a + b, 0) / arr.length; }
function std(arr) { if (arr.length < 2) return NaN; const m = mean(arr); const v = arr.reduce((acc, x) => acc + (x - m) ** 2, 0) / (arr.length - 1); return Math.sqrt(v); }
function covariance(a, b) {
  const n = Math.min(a.length, b.length); if (n < 2) return NaN;
  const ma = mean(a.slice(0, n)); const mb = mean(b.slice(0, n));
  let s = 0; for (let i = 0; i < n; i++) s += (a[i] - ma) * (b[i] - mb);
  return s / (n - 1);
}
function correlation(a, b) { return covariance(a, b) / (std(a) * std(b)); }
function rolling(arr, win, fn) { const out = []; for (let i = 0; i < arr.length; i++) { const start = Math.max(0, i - win + 1); out.push(fn(arr.slice(start, i + 1))); } return out; }
function rollingVol(returnsArr, window = 30) { return rolling(returnsArr, window, (s) => std(s) * Math.sqrt(TRADING_DAYS)); }
function rollingCorr(retsA, retsB, window = 30) { const out = []; for (let i = 0; i < retsA.length; i++) { const start = Math.max(0, i - window + 1); const a = retsA.slice(start, i + 1); const b = retsB.slice(start, i + 1); out.push(correlation(a, b)); } return out; }
function maxDrawdown(prices) { let peak = prices[0]; let maxDD = 0; for (let i = 1; i < prices.length; i++) { peak = Math.max(peak, prices[i]); maxDD = Math.min(maxDD, (prices[i] - peak) / peak); } return Math.abs(maxDD); }
function quantile(arr, q) { if (!arr.length) return NaN; const s = [...arr].sort((a, b) => a - b); const pos = (s.length - 1) * q; const base = Math.floor(pos); const rest = pos - base; return s[base + 1] !== undefined ? s[base] + rest * (s[base + 1] - s[base]) : s[base]; }
function expectedShortfall(arr, alpha = 0.05) { const s = [...arr].sort((a, b) => a - b); const cut = Math.max(1, Math.floor(s.length * alpha)); const tail = s.slice(0, cut); return tail.length ? mean(tail) : NaN; }
function toNormalizedIndex(aligned) { if (!aligned.length) return []; const p0A = aligned[0].aPrice; const p0B = aligned[0].bPrice; return aligned.map((d) => ({ date: d.date, A: (d.aPrice / p0A) * 100, B: (d.bPrice / p0B) * 100 })); }
function useInterval(callback, delay) { const saved = useRef(); useEffect(() => { saved.current = callback; }, [callback]); useEffect(() => { if (delay === null) return; const id = setInterval(() => saved.current && saved.current(), delay); return () => clearInterval(id); }, [delay]); }

// Small async helpers
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function axiosGetWithRetry(url, opts = {}, retries = 2, backoffMs = 800) {
  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      return await axios.get(url, { timeout: 20000, headers: { Accept: "application/json" }, ...opts });
    } catch (e) {
      if (attempt === retries) throw e;
      await sleep(backoffMs * (attempt + 1));
    }
  }
}

// ————————————————————————————————————————————————————————————————————
// API fetchers (CoinGecko)
async function fetchCoinHistory(coinId, days) {
  const url = `https://api.coingecko.com/api/v3/coins/${coinId}/market_chart?vs_currency=usd&days=${days}`;
  const { data } = await axiosGetWithRetry(url);
  return data.prices; // [ [ts, price], ... ]
}
async function fetchOHLC(coinId, days) {
  const d = Math.min(Math.max(days, 1), 365);
  const url = `https://api.coingecko.com/api/v3/coins/${coinId}/ohlc?vs_currency=usd&days=${d}`;
  const { data } = await axiosGetWithRetry(url);
  // [[ts, o,h,l,c], ...]
  return (data || []).map(([t, o, h, l, c]) => ({ time: Math.floor(t / 1000), open: o, high: h, low: l, close: c }));
}
async function fetchMarkets(ids) {
  const url = `https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids=${ids.join(",")}&order=market_cap_desc&per_page=${ids.length}&page=1&price_change_percentage=1h,24h,7d&sparkline=false`;
  const { data } = await axiosGetWithRetry(url);
  return data || []; // [{id,symbol,current_price,price_change_percentage_24h,...}]
}

// ————————————————————————————————————————————————————————————————————
// Small components
function MetricCard({ title, value, format = (x) => x, big = false }) {
  return (
    <div className={`bg-white rounded-2xl shadow p-4 ${big ? "md:col-span-1" : ""}`}>
      <div className="text-xs text-gray-500 mb-1">{title}</div>
      <div className={`font-semibold ${big ? "text-2xl" : "text-xl"}`}>{value === null || value === undefined ? "—" : format(value)}</div>
    </div>
  );
}
function ChartCard({ title, right, children }) {
  return (
    <div className="bg-white rounded-2xl shadow p-4">
      <div className="flex items-center justify-between mb-2">
        <div className="text-sm font-semibold">{title}</div>
        {right}
      </div>
      {children}
    </div>
  );
}
function Pill({ children }) { return <span className="px-2 py-0.5 rounded-full bg-gray-100 text-gray-700 text-xs">{children}</span>; }

// Candles component with dynamic import + fallback line chart
function Candles({ data }) {
  const ref = useRef(null);
  const [fallback, setFallback] = useState(false);

  const lineData = useMemo(() => (data || []).map((d) => ({ date: toISODateTime(d.time), close: d.close })), [data]);

  useEffect(() => {
    let chart;
    let cleanup;
    (async () => {
      if (!ref.current) return;
      try {
        const mod = await import('lightweight-charts');
        const create = mod.createChart || (mod.default && mod.default.createChart);
        if (!create) {
          setFallback(true);
          return;
        }
        chart = create(ref.current, {
          width: ref.current.clientWidth,
          height: 280,
          rightPriceScale: { borderVisible: false },
          timeScale: { borderVisible: false },
          grid: { horzLines: { color: '#eee' }, vertLines: { color: '#f7f7f7' } },
        });
        if (typeof chart.addCandlestickSeries !== 'function') {
          setFallback(true);
          chart.remove();
          chart = undefined;
          return;
        }
        const series = chart.addCandlestickSeries();
        series.setData(data || []);
        const onResize = () => chart && chart.applyOptions({ width: ref.current.clientWidth });
        window.addEventListener('resize', onResize);
        cleanup = () => { window.removeEventListener('resize', onResize); chart && chart.remove(); };
      } catch (e) {
        console.error('Candles fallback due to error:', e);
        setFallback(true);
      }
    })();
    return () => { cleanup && cleanup(); };
  }, [data]);

  if (fallback) {
    return (
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={lineData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis domain={["auto", "auto"]} />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="close" name="Clôture" dot={false} strokeWidth={2} />
        </LineChart>
      </ResponsiveContainer>
    );
  }

  return <div ref={ref} className="w-full" />;
}

// ————————————————————————————————————————————————————————————————————
export default function ProDualCryptoDashboard() {
  // Selection & params
  const [coinA, setCoinA] = useState("bitcoin");
  const [coinB, setCoinB] = useState("ethereum");
  const [days, setDays] = useState(365);
  const [corWin, setCorWin] = useState(30);
  const [notional, setNotional] = useState(10000);
  const [riskFree, setRiskFree] = useState(0.02);

  // States
  const [loadingHist, setLoadingHist] = useState(false);
  const [errorHist, setErrorHist] = useState(null);
  const [alignedSeries, setAlignedSeries] = useState([]); // {date,aPrice,bPrice}
  const [ohlcA, setOhlcA] = useState([]);
  const [ohlcB, setOhlcB] = useState([]);
  const [markets, setMarkets] = useState([]);
  const [lastTicksAt, setLastTicksAt] = useState(null);
  const [lastHistAt, setLastHistAt] = useState(null);

  // Fetch historical + OHLC
  const loadHist = async () => {
    try {
      setLoadingHist(true); setErrorHist(null);
      const [a, b, oA, oB] = await Promise.all([
        fetchCoinHistory(coinA, days),
        fetchCoinHistory(coinB, days),
        fetchOHLC(coinA, Math.min(days, 180)),
        fetchOHLC(coinB, Math.min(days, 180)),
      ]);
      const aMap = seriesToDailyMap(a); const bMap = seriesToDailyMap(b);
      const commonDates = intersectSorted(aMap.keys(), bMap.keys());
      const aligned = commonDates.map((d) => ({ date: d, aPrice: aMap.get(d), bPrice: bMap.get(d) }));
      setAlignedSeries(aligned); setOhlcA(oA); setOhlcB(oB); setLastHistAt(new Date());
    } catch (e) {
      console.error(e);
      setErrorHist("Erreur de chargement des historiques (réseau/limite API). Réessaye ou réduis la fenêtre.");
    } finally { setLoadingHist(false); }
  };
  useEffect(() => { loadHist(); }, [coinA, coinB, days]);
  useInterval(() => { loadHist(); }, REFRESH_MS_HIST);

  // Fetch markets (live prices)
  const loadTicks = async () => {
    try {
      const ids = COINS.map((c) => c.id);
      const data = await fetchMarkets(ids);
      setMarkets(data); setLastTicksAt(new Date());
    } catch (e) { console.error(e); }
  };
  useEffect(() => { loadTicks(); }, []);
  useInterval(() => { loadTicks(); }, REFRESH_MS_TICKS);

  // Derived: metrics pair
  const metrics = useMemo(() => {
    if (alignedSeries.length < 3) return null;
    const pricesA = alignedSeries.map((d) => d.aPrice);
    const pricesB = alignedSeries.map((d) => d.bPrice);
    const retsA = computeLogReturns(pricesA);
    const retsB = computeLogReturns(pricesB);

    const muA_d = mean(retsA), muB_d = mean(retsB);
    const varA_d = std(retsA) ** 2, varB_d = std(retsB) ** 2;
    const volA_ann = Math.sqrt(varA_d) * Math.sqrt(TRADING_DAYS);
    const volB_ann = Math.sqrt(varB_d) * Math.sqrt(TRADING_DAYS);
    const muA_ann = muA_d * TRADING_DAYS, muB_ann = muB_d * TRADING_DAYS;

    const corrAB = correlation(retsA, retsB);
    const covAB = covariance(retsA, retsB);
    const betaAonB = covAB / varB_d;
    const r2 = (corrAB ** 2);

    const mddA = maxDrawdown(pricesA); const mddB = maxDrawdown(pricesB);
    const sharpeA = (muA_ann - riskFree) / volA_ann; const sharpeB = (muB_ann - riskFree) / volB_ann;

    const alpha = 0.05;
    const varA = -quantile(retsA, alpha) * notional; const varB = -quantile(retsB, alpha) * notional;
    const esA = -expectedShortfall(retsA, alpha) * notional; const esB = -expectedShortfall(retsB, alpha) * notional;

    // Rolling
    const rollVolA = rollingVol(retsA, 30), rollVolB = rollingVol(retsB, 30);
    const rollCorr = rollingCorr(retsA, retsB, Math.max(5, Math.min(corWin, 120)));

    // Regression line for scatter (y = a + b x)
    const mx = mean(retsA), my = mean(retsB);
    const b = betaAonB; const a = my - b * mx;
    const minX = Math.min(...retsA), maxX = Math.max(...retsA);
    const regLine = [{ x: minX, y: a + b * minX }, { x: maxX, y: a + b * maxX }];
    const scatter = retsA.map((ra, i) => ({ x: ra, y: retsB[i] }));

    // Efficient frontier 2-actifs & optimal weights
    const frontier = [];
    let wMinVar = 0.5; let minVar = Infinity; let wMaxSharpe = 0.5; let bestSharpe = -Infinity;
    for (let w = 0; w <= 1.0001; w += 0.02) {
      const mu = w * muA_ann + (1 - w) * muB_ann;
      const sigma = Math.sqrt(w ** 2 * varA_d * TRADING_DAYS + (1 - w) ** 2 * varB_d * TRADING_DAYS + 2 * w * (1 - w) * covAB * TRADING_DAYS);
      frontier.push({ w: Number(w.toFixed(2)), mu, sigma });
      if (sigma < minVar) { minVar = sigma; wMinVar = w; }
      const sh = (mu - riskFree) / sigma; if (isFinite(sh) && sh > bestSharpe) { bestSharpe = sh; wMaxSharpe = w; }
    }

    return { muA_ann, muB_ann, volA_ann, volB_ann, corrAB, covAB, betaAonB, r2, mddA, mddB, sharpeA, sharpeB, varA, varB, esA, esB, rollVolA, rollVolB, rollCorr, scatter, regLine, retsA, retsB, frontier, wMinVar, wMaxSharpe };
  }, [alignedSeries, riskFree, notional, corWin]);

  const priceIndexSeries = useMemo(() => alignedSeries.length < 2 ? [] : toNormalizedIndex(alignedSeries), [alignedSeries]);
  const volSeries = useMemo(() => {
    if (!metrics || alignedSeries.length < 2) return [];
    const dates = alignedSeries.slice(1).map((d) => d.date);
    return dates.map((date, i) => ({ date, A: metrics.rollVolA[i], B: metrics.rollVolB[i] }));
  }, [alignedSeries, metrics]);
  const rollCorrSeries = useMemo(() => {
    if (!metrics || alignedSeries.length < 2) return [];
    const dates = alignedSeries.slice(1).map((d) => d.date);
    return dates.map((date, i) => ({ date, corr: metrics.rollCorr[i] }));
  }, [alignedSeries, metrics]);

  const coinAObj = COINS.find((c) => c.id === coinA);
  const coinBObj = COINS.find((c) => c.id === coinB);
  const mktA = markets.find((m) => m.id === coinA);
  const mktB = markets.find((m) => m.id === coinB);

  // Export CSV
  function csvDownload(aligned) {
    if (!aligned.length) return;
    const pricesA = aligned.map((d) => d.aPrice);
    const pricesB = aligned.map((d) => d.bPrice);
    const retsA = computeLogReturns(pricesA);
    const retsB = computeLogReturns(pricesB);
    const rows = ["date,priceA,priceB,retA,retB"]; // ret columns start from t1
    for (let i = 0; i < aligned.length; i++) {
      const d = aligned[i]; const rA = i === 0 ? "" : retsA[i - 1]; const rB = i === 0 ? "" : retsB[i - 1];
      rows.push(`${d.date},${d.aPrice},${d.bPrice},${rA},${rB}`);
    }
    const blob = new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob); const a = document.createElement("a");
    a.href = url; a.download = "dual_crypto_series.csv"; a.click(); URL.revokeObjectURL(url);
  }

  // ————————————————— Diagnostics (tests élargis) —————————————————
  const diagnostics = useMemo(() => {
    const cases = [];
    // Test 1: log return simple
    try {
      const ret = computeLogReturns([100, 110])[0];
      const pass = Math.abs(ret - Math.log(1.1)) < 1e-12;
      cases.push({ name: "LogReturn 100→110", pass, got: ret, want: Math.log(1.1) });
    } catch (e) { cases.push({ name: "LogReturn 100→110", pass: false, got: e.message, want: "≈0.09531" }); }
    // Test 2: correlation perfect
    try {
      const c = correlation([1,2,3],[2,4,6]);
      const pass = Math.abs(c - 1) < 1e-12;
      cases.push({ name: "Corr perfect", pass, got: c, want: 1 });
    } catch (e) { cases.push({ name: "Corr perfect", pass: false, got: e.message, want: 1 }); }
    // Test 3: quantile median
    try {
      const q = quantile([1,2,3,4,5], 0.5);
      const pass = q === 3;
      cases.push({ name: "Quantile median", pass, got: q, want: 3 });
    } catch (e) { cases.push({ name: "Quantile median", pass: false, got: e.message, want: 3 }); }
    // Test 4: ES 20% on sorted negatives
    try {
      const es = expectedShortfall([-3,-2,-1,0,1], 0.2);
      const pass = es === -3;
      cases.push({ name: "ES alpha=0.2", pass, got: es, want: -3 });
    } catch (e) { cases.push({ name: "ES alpha=0.2", pass: false, got: e.message, want: -3 }); }
    // Test 5: std known vector
    try {
      const s = std([1,2,3,4]);
      const pass = Math.abs(s - Math.sqrt(5/3)) < 1e-12; // sample std
      cases.push({ name: "Std [1,2,3,4]", pass, got: s, want: Math.sqrt(5/3) });
    } catch (e) { cases.push({ name: "Std [1,2,3,4]", pass: false, got: e.message, want: "≈1.29099" }); }
    // Test 6: max drawdown simple
    try {
      const dd = maxDrawdown([100,120,90,95]);
      const pass = Math.abs(dd - 0.25) < 1e-12;
      cases.push({ name: "MaxDD 100,120,90,95", pass, got: dd, want: 0.25 });
    } catch (e) { cases.push({ name: "MaxDD 100,120,90,95", pass: false, got: e.message, want: 0.25 }); }
    // Test 7: covariance vs corr
    try {
      const a = [1,2,3,4,5]; const b = [2,1,0,-1,-2];
      const cov = covariance(a,b); const corr = correlation(a,b); const s = std(a)*std(b);
      const pass = Math.abs(cov - corr * s) < 1e-12;
      cases.push({ name: "Cov = Corr*σaσb", pass, got: cov, want: corr*s });
    } catch (e) { cases.push({ name: "Cov = Corr*σaσb", pass: false, got: e.message, want: "identity" }); }
    // Test 8: normalized index start
    try {
      const idx = toNormalizedIndex([{date:'d1', aPrice:10, bPrice:20}]);
      const pass = Math.abs(idx[0].A - 100) < 1e-12 && Math.abs(idx[0].B - 100) < 1e-12;
      cases.push({ name: "Index base 100", pass, got: JSON.stringify(idx[0]), want: '{A:100,B:100}' });
    } catch (e) { cases.push({ name: "Index base 100", pass: false, got: e.message, want: "A=B=100" }); }
    return cases;
  }, []);

  return (
    <div className="min-h-screen w-full p-6 bg-gray-50 text-gray-900">
      <div className="max-w-7xl mx-auto">
        {/* Header & live ticker */}
        <header className="mb-4">
          <h1 className="text-2xl md:text-3xl font-bold">Dual Crypto — Pro Correlation, Risk & Vol Workbench</h1>
          <p className="text-xs text-gray-600 mt-1">Source: CoinGecko • Hist MàJ 10 min • Ticks MàJ 60 s • Rendements log, 252 j/an</p>
        </header>

        {/* Diagnostics */}
        <ChartCard title="Diagnostics (tests de base)">
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-2">
            {diagnostics.map((t, i) => (
              <div key={i} className={`rounded-xl border p-2 ${t.pass ? 'border-green-300 bg-green-50' : 'border-red-300 bg-red-50'}`}>
                <div className="text-xs font-semibold">{t.name}</div>
                <div className="text-[11px]">Résultat: {String(t.got)}</div>
                <div className="text-[11px] text-gray-600">Attendu: {String(t.want)}</div>
                <div className={`text-[11px] ${t.pass ? 'text-green-700' : 'text-red-700'}`}>{t.pass ? 'PASS' : 'FAIL'}</div>
              </div>
            ))}
          </div>
        </ChartCard>

        {/* Live ticker */}
        <div className="overflow-x-auto whitespace-nowrap mb-4">
          <div className="inline-flex gap-2">
            {(markets || []).slice(0, 12).map((m) => (
              <div key={m.id} className="px-3 py-2 rounded-xl bg-white shadow inline-flex items-center gap-2">
                <span className="font-semibold">{m.symbol?.toUpperCase()}</span>
                <span>${fmtNum(m.current_price, 2)}</span>
                <span className={`text-xs ${m.price_change_percentage_24h >= 0 ? 'text-green-600' : 'text-red-600'}`}>{m.price_change_percentage_24h?.toFixed(2)}%</span>
                <span className="text-[10px] text-gray-500">MC {fmtNum(m.market_cap)}</span>
              </div>
            ))}
          </div>
          <div className="text-[10px] text-gray-500 mt-1">Dernière MàJ prix: {lastTicksAt ? lastTicksAt.toLocaleTimeString() : '—'}</div>
        </div>

        {/* Controls */}
        <div className="grid md:grid-cols-6 gap-3 mb-6">
          <div className="bg-white rounded-2xl shadow p-4">
            <label className="block text-xs text-gray-500 mb-1">Actif A</label>
            <select className="w-full border rounded-xl p-2" value={coinA} onChange={(e) => setCoinA(e.target.value)}>
              {COINS.map((c) => <option key={c.id} value={c.id}>{c.symbol} — {c.id}</option>)}
            </select>
            <div className="mt-1 text-xs text-gray-600">Spot: {mktA ? `$${fmtNum(mktA.current_price,2)}` : '—'}</div>
          </div>
          <div className="bg-white rounded-2xl shadow p-4">
            <label className="block text-xs text-gray-500 mb-1">Actif B</label>
            <select className="w-full border rounded-xl p-2" value={coinB} onChange={(e) => setCoinB(e.target.value)}>
              {COINS.map((c) => <option key={c.id} value={c.id}>{c.symbol} — {c.id}</option>)}
            </select>
            <button onClick={() => { const a = coinA; setCoinA(coinB); setCoinB(a); }} className="mt-2 w-full border rounded-xl py-1">↕️ Inverser</button>
            <div className="mt-1 text-xs text-gray-600">Spot: {mktB ? `$${fmtNum(mktB.current_price,2)}` : '—'}</div>
          </div>
          <div className="bg-white rounded-2xl shadow p-4">
            <label className="block text-xs text-gray-500 mb-1">Fenêtre historique</label>
            <div className="flex gap-2">
              {DAY_OPTIONS.map((opt) => (
                <button key={opt.value} onClick={() => setDays(opt.value)} className={`flex-1 px-3 py-2 rounded-xl border ${days === opt.value ? "bg-gray-900 text-white" : "bg-white"}`}>{opt.label}</button>
              ))}
            </div>
            <div className="mt-3">
              <label className="block text-xs text-gray-500 mb-1">Fenêtre corr. roulante: {corWin} j</label>
              <input type="range" min={5} max={120} value={corWin} onChange={(e) => setCorWin(Number(e.target.value))} className="w-full" />
            </div>
          </div>
          <div className="bg-white rounded-2xl shadow p-4 grid grid-cols-2 gap-3 md:col-span-3">
            <div>
              <label className="block text-xs text-gray-500 mb-1">Notional ($)</label>
              <input type="number" className="w-full border rounded-xl p-2" value={notional} onChange={(e) => setNotional(Number(e.target.value))} />
            </div>
            <div>
              <label className="block text-xs text-gray-500 mb-1">Taux sans risque (annuel)</label>
              <input title="Ex: 0.02 = 2%" type="number" step="0.001" className="w-full border rounded-xl p-2" value={riskFree} onChange={(e) => setRiskFree(Number(e.target.value))} />
            </div>
            <div className="col-span-2 flex gap-2">
              <button onClick={() => csvDownload(alignedSeries)} className="px-3 py-2 rounded-xl border w-full">Télécharger CSV</button>
              <button onClick={loadHist} className="px-3 py-2 rounded-xl border w-full">Rafraîchir</button>
            </div>
          </div>
        </div>

        {/* Status */}
        <div className="flex items-center justify-between mb-4 text-sm">
          <div>
            {loadingHist ? <span className="text-gray-600">Chargement…</span> : errorHist ? <span className="text-red-600">{errorHist}</span> : lastHistAt ? <span className="text-gray-600">Hist MàJ: {lastHistAt.toLocaleString()}</span> : null}
          </div>
          <div className="text-gray-600">Pair: <strong>{coinAObj?.symbol}/{coinBObj?.symbol}</strong> • Fenêtre: <strong>{days} j</strong></div>
        </div>

        {/* KPI cards */}
        <div className="grid md:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
          <MetricCard title={`Corr (${coinAObj?.symbol}, ${coinBObj?.symbol})`} value={metrics?.corrAB} format={(x) => fmtPct(x, 2)} />
          <MetricCard title={`Bêta (${coinAObj?.symbol} sur ${coinBObj?.symbol})`} value={metrics?.betaAonB} format={(x) => isFinite(x) ? x.toFixed(2) : '—'} />
          <MetricCard title={`R²`} value={metrics?.r2} format={(x) => (isFinite(x) ? x.toFixed(2) : '—')} />
          <MetricCard title={`Vol ${coinAObj?.symbol} (ann)`} value={metrics?.volA_ann} format={(x) => fmtPct(x, 2)} />
          <MetricCard title={`Vol ${coinBObj?.symbol} (ann)`} value={metrics?.volB_ann} format={(x) => fmtPct(x, 2)} />
          <MetricCard title={`Sharpe ${coinAObj?.symbol}`} value={metrics?.sharpeA} format={(x) => (isFinite(x) ? x.toFixed(2) : '—')} />
        </div>
        <div className="grid md:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
          <MetricCard title={`Sharpe ${coinBObj?.symbol}`} value={metrics?.sharpeB} format={(x) => (isFinite(x) ? x.toFixed(2) : '—')} />
          <MetricCard title={`Max DD ${coinAObj?.symbol}`} value={metrics?.mddA} format={(x) => fmtPct(x, 1)} />
          <MetricCard title={`Max DD ${coinBObj?.symbol}`} value={metrics?.mddB} format={(x) => fmtPct(x, 1)} />
          <MetricCard title={`VaR 1j 95% ${coinAObj?.symbol}`} value={metrics?.varA} format={(x) => `$${fmtNum(x, 0)}`} />
          <MetricCard title={`VaR 1j 95% ${coinBObj?.symbol}`} value={metrics?.varB} format={(x) => `$${fmtNum(x, 0)}`} />
          <MetricCard title={`ES 1j 95% ${coinAObj?.symbol}/${coinBObj?.symbol}`} value={metrics ? (metrics.esA + metrics.esB) / 2 : null} format={(x) => `$${fmtNum(x, 0)}`} />
        </div>

        {/* Price & Vol */}
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <ChartCard title="Indice de Prix Normalisé (départ = 100)">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={priceIndexSeries} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 12 }} domain={["auto", "auto"]} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="A" name={`A: ${coinAObj?.symbol}`} dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="B" name={`B: ${coinBObj?.symbol}`} dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Volatilité Glissante 30j (annualisée)">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={volSeries} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 12 }} />
                <YAxis tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                <Tooltip formatter={(v) => fmtPct(v, 2)} />
                <Legend />
                <Line type="monotone" dataKey="A" name={`A: ${coinAObj?.symbol}`} dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="B" name={`B: ${coinBObj?.symbol}`} dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>

        {/* Rolling Corr & Regression */}
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <ChartCard title={`Corrélation roulante (${corWin} j)`}>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={rollCorrSeries}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="date" />
                <YAxis domain={[-1, 1]} />
                <Tooltip formatter={(v) => fmtPct(v, 2)} />
                <Legend />
                <Line type="monotone" dataKey="corr" name="Corr" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard title="Nuage ret(A) vs ret(B) + droite de régression">
            <ResponsiveContainer width="100%" height={300}>
              <ComposedChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="x" type="number" tickFormatter={(v) => `${(v * 100).toFixed(1)}%`} />
                <YAxis dataKey="y" type="number" tickFormatter={(v) => `${(v * 100).toFixed(1)}%`} />
                <Tooltip formatter={(v) => `${(v * 100).toFixed(2)}%`} />
                <Legend />
                <Scatter data={metrics?.scatter || []} name="Points" />
                <Line data={metrics?.regLine || []} dataKey="y" name="Régression" dot={false} strokeWidth={2} xAxisId={0} yAxisId={0} />
              </ComposedChart>
            </ResponsiveContainer>
            <div className="text-xs text-gray-500 mt-2">Bêta(A/B) {metrics?.betaAonB?.toFixed ? metrics.betaAonB.toFixed(2) : '—'} • R² {metrics?.r2?.toFixed ? metrics.r2.toFixed(2) : '—'}</div>
          </ChartCard>
        </div>

        {/* Efficient Frontier */}
        <ChartCard title="Frontière efficiente 2-actifs (annualisée)" right={<Pill>Taux sans risque {fmtPct(riskFree,2)}</Pill>}>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="sigma" name="Risque (σ)" tickFormatter={(v) => fmtPct(v,0)} />
              <YAxis dataKey="mu" name="Rendement (μ)" tickFormatter={(v) => fmtPct(v,0)} />
              <Tooltip formatter={(v) => fmtPct(v,2)} />
              <Legend />
              <Line data={metrics?.frontier || []} dataKey="mu" name="Frontière" dot={false} strokeWidth={2} />
              <Scatter data={metrics?.frontier || []} name="Portfolios" />
            </ScatterChart>
          </ResponsiveContainer>
          <div className="text-xs text-gray-600 mt-2">Poids min-variance A: <strong>{metrics ? fmtPctAbs(metrics.wMinVar * 100) : '—'}</strong> • Poids max-Sharpe A: <strong>{metrics ? fmtPctAbs(metrics.wMaxSharpe * 100) : '—'}</strong></div>
        </ChartCard>

        {/* Candles */}
        <div className="grid md:grid-cols-2 gap-6 mb-6">
          <ChartCard title={`Chandeliers ${coinAObj?.symbol} (≈${Math.min(days,180)} j)`}>
            <Candles data={ohlcA} />
          </ChartCard>
          <ChartCard title={`Chandeliers ${coinBObj?.symbol} (≈${Math.min(days,180)} j)`}>
            <Candles data={ohlcB} />
          </ChartCard>
        </div>

        {/* Markets Table */}
        <ChartCard title="Prix du marché (watchlist) - spot & variations">
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="text-left text-gray-500">
                  <th className="py-2 pr-4">Actif</th>
                  <th className="py-2 pr-4">Prix</th>
                  <th className="py-2 pr-4">1h</th>
                  <th className="py-2 pr-4">24h</th>
                  <th className="py-2 pr-4">7j</th>
                  <th className="py-2 pr-4">Volume 24h</th>
                  <th className="py-2 pr-4">Market Cap</th>
                </tr>
              </thead>
              <tbody>
                {(markets || []).map((m) => (
                  <tr key={m.id} className="border-t">
                    <td className="py-2 pr-4 font-medium">{m.name} <span className="text-gray-500">({m.symbol?.toUpperCase()})</span></td>
                    <td className="py-2 pr-4">${fmtNum(m.current_price, 4)}</td>
                    <td className={`py-2 pr-4 ${m.price_change_percentage_1h_in_currency >= 0 ? 'text-green-600' : 'text-red-600'}`}>{m.price_change_percentage_1h_in_currency?.toFixed(2)}%</td>
                    <td className={`py-2 pr-4 ${m.price_change_percentage_24h_in_currency >= 0 ? 'text-green-600' : 'text-red-600'}`}>{m.price_change_percentage_24h_in_currency?.toFixed(2)}%</td>
                    <td className={`py-2 pr-4 ${m.price_change_percentage_7d_in_currency >= 0 ? 'text-green-600' : 'text-red-600'}`}>{m.price_change_percentage_7d_in_currency?.toFixed(2)}%</td>
                    <td className="py-2 pr-4">${fmtNum(m.total_volume)}</td>
                    <td className="py-2 pr-4">${fmtNum(m.market_cap)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="text-[10px] text-gray-500 mt-1">Dernière MàJ prix: {lastTicksAt ? lastTicksAt.toLocaleTimeString() : '—'}</div>
          </div>
        </ChartCard>

        <footer className="text-[11px] text-gray-500 mt-6">
          <p>
            Méthodo: rendements log journaliers; vol annualisée 252 jours; corrélation de Pearson; bêta via OLS (retsB sur retsA). VaR & ES (CVaR) 95% empiriques (historical simulation).
            Données publiques CoinGecko (peuvent être limitées en fréquence). Outils à visée d'analyse uniquement, pas de conseil en investissement.
          </p>
        </footer>
      </div>
    </div>
  );
}

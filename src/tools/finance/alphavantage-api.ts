import type { ApiResponse } from './api.js';
import { MissingProviderApiKeyError } from './errors.js';

const BASE_URL = 'https://www.alphavantage.co/query';

const DEFAULT_MIN_INTERVAL_MS = 1100;

let lastAlphaVantageCallAt = 0;
let alphaVantageQueue: Promise<void> = Promise.resolve();

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function getMinIntervalMs(): number {
  const raw = process.env.ALPHAVANTAGE_MIN_INTERVAL_MS;
  if (!raw) return DEFAULT_MIN_INTERVAL_MS;
  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_MIN_INTERVAL_MS;
}

async function applyAlphaVantageRateLimit(): Promise<void> {
  const minIntervalMs = getMinIntervalMs();
  const now = Date.now();
  const waitMs = Math.max(0, lastAlphaVantageCallAt + minIntervalMs - now);
  if (waitMs > 0) {
    await sleep(waitMs);
  }
  lastAlphaVantageCallAt = Date.now();
}

function toAlphaVantageDateTime(date: string, time: string): string {
  // date: YYYY-MM-DD, time: HHmm
  const compact = date.replace(/-/g, '');
  return `${compact}T${time}`;
}

function buildRateLimitGuidance(providerMessage?: string): string {
  const details = providerMessage ? ` Details: ${providerMessage}` : '';
  return (
    `Alpha Vantage rate limit hit.${details} ` +
    `Guidance: reduce tool calls, batch questions, or wait and retry. ` +
    `You can also switch providers by setting FINANCE_PROVIDERS=financialdatasets (or FINANCE_PROVIDER=financialdatasets).`
  );
}

export interface AlphaVantageOptions {
  /** When provided, maps to Alpha Vantage 'outputsize' for time series endpoints. */
  outputsize?: 'compact' | 'full';
}

export async function callAlphaVantage(
  avFunction: string,
  params: Record<string, string | number | undefined>,
  options?: AlphaVantageOptions
): Promise<ApiResponse> {
  const apiKey = process.env.ALPHAVANTAGE_API_KEY;
  if (!apiKey || !apiKey.trim() || apiKey.trim().startsWith('your-')) {
    throw new MissingProviderApiKeyError(
      'alphavantage',
      'ALPHAVANTAGE_API_KEY is required when using the alphavantage provider. Set it in your .env file.'
    );
  }

  const url = new URL(BASE_URL);
  url.searchParams.set('function', avFunction);
  url.searchParams.set('apikey', apiKey);

  if (options?.outputsize) {
    url.searchParams.set('outputsize', options.outputsize);
  }

  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null) {
      url.searchParams.set(key, String(value));
    }
  }

  // Enforce provider rate limits (Alpha Vantage throttles aggressively).
  // This also makes parallel tool calls (Promise.all) behave safely.
  alphaVantageQueue = alphaVantageQueue.then(applyAlphaVantageRateLimit, applyAlphaVantageRateLimit);
  await alphaVantageQueue;

  const response = await fetch(url.toString());

  // Alpha Vantage often returns 200 with a JSON "Note" when throttled, but also handle non-OK.
  let data: unknown;
  try {
    data = await response.json();
  } catch (e) {
    throw new Error(`Alpha Vantage request failed to parse JSON. HTTP ${response.status}.`);
  }

  if (!response.ok) {
    // Try to surface provider message if present.
    const msg =
      data &&
      typeof data === 'object' &&
      ('Note' in data || 'Information' in data || 'Error Message' in data)
        ? String((data as Record<string, unknown>)['Note'] ?? (data as Record<string, unknown>)['Information'] ?? (data as Record<string, unknown>)['Error Message'])
        : undefined;

    if (response.status === 429) {
      throw new Error(buildRateLimitGuidance(msg));
    }

    throw new Error(`Alpha Vantage API request failed: ${response.status} ${response.statusText}${msg ? ` (${msg})` : ''}`);
  }

  // Rate-limit / error signals in successful responses.
  if (data && typeof data === 'object') {
    const obj = data as Record<string, unknown>;

    if (typeof obj['Error Message'] === 'string' && obj['Error Message'].trim()) {
      throw new Error(`Alpha Vantage error: ${obj['Error Message']}`);
    }

    if (typeof obj['Note'] === 'string' && obj['Note'].trim()) {
      throw new Error(buildRateLimitGuidance(obj['Note']));
    }

    if (typeof obj['Information'] === 'string' && obj['Information'].trim()) {
      // Alpha Vantage sometimes uses Information for limit/service messages.
      const info = obj['Information'] as string;
      if (info.toLowerCase().includes('limit') || info.toLowerCase().includes('frequency')) {
        throw new Error(buildRateLimitGuidance(info));
      }
    }
  }

  return { data: (data ?? {}) as Record<string, unknown>, url: url.toString() };
}

export function buildAlphaVantageNewsParams(input: {
  ticker: string;
  start_date?: string;
  end_date?: string;
  limit: number;
}): Record<string, string | number | undefined> {
  return {
    tickers: input.ticker.toUpperCase(),
    time_from: input.start_date ? toAlphaVantageDateTime(input.start_date, '0000') : undefined,
    time_to: input.end_date ? toAlphaVantageDateTime(input.end_date, '2359') : undefined,
    limit: input.limit,
    sort: 'LATEST',
  };
}

export function parseCryptoTickerPair(ticker: string): { from: string; to: string } {
  const parts = ticker.split('-').map(p => p.trim()).filter(Boolean);
  if (parts.length !== 2) {
    throw new Error(
      `Invalid crypto ticker format: '${ticker}'. Expected 'CRYPTO-USD' (e.g., 'BTC-USD').`
    );
  }
  return { from: parts[0].toUpperCase(), to: parts[1].toUpperCase() };
}

export function toIntradayInterval(multiplier: number): string {
  const allowed = new Set([1, 5, 15, 30, 60]);
  if (!allowed.has(multiplier)) {
    throw new Error(
      `Alpha Vantage intraday interval only supports multipliers {1,5,15,30,60}. Got ${multiplier}.`
    );
  }
  return `${multiplier}min`;
}

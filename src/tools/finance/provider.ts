import {
  type FinanceProviderName,
  UnsupportedByProviderError,
  isMissingProviderApiKeyError,
  isUnsupportedByProviderError,
} from './errors.js';

function parseFinanceProviderName(raw: string): FinanceProviderName | null {
  const normalized = raw.trim().toLowerCase();
  if (!normalized) return null;

  if (
    normalized === 'alphavantage' ||
    normalized === 'alpha-vantage' ||
    normalized === 'alpha_vantage' ||
    normalized === 'alpha'
  ) {
    return 'alphavantage';
  }

  if (
    normalized === 'financialdatasets' ||
    normalized === 'financial-datasets' ||
    normalized === 'financial_datasets' ||
    normalized === 'fd'
  ) {
    return 'financialdatasets';
  }

  if (
    normalized === 'fmp' ||
    normalized === 'financialmodelingprep' ||
    normalized === 'financial-modeling-prep' ||
    normalized === 'financial_modeling_prep'
  ) {
    return 'fmp';
  }

  return null;
}

export function getFinanceProviders(): FinanceProviderName[] {
  const rawList = process.env.FINANCE_PROVIDERS;
  if (rawList && rawList.trim()) {
    const providers: FinanceProviderName[] = [];
    for (const part of rawList.split(',')) {
      const parsed = parseFinanceProviderName(part);
      if (parsed && !providers.includes(parsed)) {
        providers.push(parsed);
      }
    }
    if (providers.length > 0) return providers;
  }

  const single = process.env.FINANCE_PROVIDER;
  if (single && single.trim()) {
    const parsed = parseFinanceProviderName(single);
    if (parsed) return [parsed];
  }

  return ['financialdatasets'];
}

// Backward-compatible helper used by older code paths.
export function getFinanceProviderName(): FinanceProviderName {
  return getFinanceProviders()[0] ?? 'financialdatasets';
}

export function unsupportedByProvider(provider: FinanceProviderName, toolName: string): Error {
  return new UnsupportedByProviderError(
    provider,
    toolName,
    `${toolName} is not supported by provider '${provider}'. ` +
      `Guidance: set FINANCE_PROVIDERS to include a compatible provider (e.g. 'financialdatasets,fmp,alphavantage').`
  );
}

export async function runWithFinanceProviderFallback<T>(
  toolName: string,
  run: (provider: FinanceProviderName) => Promise<T>
): Promise<T> {
  const providers = getFinanceProviders();
  const errors: Array<{ provider: FinanceProviderName; error: unknown }> = [];

  for (const provider of providers) {
    try {
      return await run(provider);
    } catch (err) {
      errors.push({ provider, error: err });

      // Continue trying other providers on any error.
      // (unsupported, throttled/rate-limited, missing key, transient HTTP errors, etc.)
      continue;
    }
  }

  const details = errors
    .map(({ provider, error }) => {
      const msg = error instanceof Error ? error.message : String(error);
      const tag = isUnsupportedByProviderError(error)
        ? 'unsupported'
        : isMissingProviderApiKeyError(error)
          ? 'missing_api_key'
          : 'error';
      return `${provider} (${tag}): ${msg}`;
    })
    .join(' | ');

  throw new Error(
    `All configured finance providers failed for tool '${toolName}'. ` +
      `Tried: ${providers.join(', ')}. ` +
      (details ? `Details: ${details}` : '')
  );
}

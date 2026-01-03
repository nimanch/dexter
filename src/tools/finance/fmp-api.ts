import type { ApiResponse } from './api.js';
import { MissingProviderApiKeyError } from './errors.js';

const BASE_URL = 'https://financialmodelingprep.com/stable';

function cleanEndpoint(endpoint: string): string {
  if (!endpoint) return '';
  return endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
}

export async function callFmp(
  endpoint: string,
  params: Record<string, string | number | undefined>
): Promise<ApiResponse> {
  const apiKey = process.env.FMP_API_KEY;
  if (!apiKey || !apiKey.trim() || apiKey.trim().startsWith('your-')) {
    throw new MissingProviderApiKeyError(
      'fmp',
      'FMP_API_KEY is required when using the fmp provider. Set it in your .env file.'
    );
  }

  const url = new URL(`${BASE_URL}${cleanEndpoint(endpoint)}`);

  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null) {
      url.searchParams.set(key, String(value));
    }
  }

  url.searchParams.set('apikey', apiKey);

  const response = await fetch(url.toString());
  if (!response.ok) {
    throw new Error(`FMP API request failed: ${response.status} ${response.statusText}`);
  }

  const data = (await response.json()) as Record<string, unknown>;
  return { data, url: url.toString() };
}

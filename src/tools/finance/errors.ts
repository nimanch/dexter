export type FinanceProviderName = 'financialdatasets' | 'alphavantage' | 'fmp';

export class UnsupportedByProviderError extends Error {
  public readonly name = 'UnsupportedByProviderError';
  public readonly provider: FinanceProviderName;
  public readonly toolName: string;

  constructor(provider: FinanceProviderName, toolName: string, message?: string) {
    super(
      message ??
        `${toolName} is not supported by provider '${provider}'.`
    );
    this.provider = provider;
    this.toolName = toolName;
  }
}

export class MissingProviderApiKeyError extends Error {
  public readonly name = 'MissingProviderApiKeyError';
  public readonly provider: FinanceProviderName;

  constructor(provider: FinanceProviderName, message?: string) {
    super(message ?? `Missing API key for provider '${provider}'.`);
    this.provider = provider;
  }
}

export function isUnsupportedByProviderError(err: unknown): err is UnsupportedByProviderError {
  return err instanceof Error && (err as any).name === 'UnsupportedByProviderError';
}

export function isMissingProviderApiKeyError(err: unknown): err is MissingProviderApiKeyError {
  return err instanceof Error && (err as any).name === 'MissingProviderApiKeyError';
}

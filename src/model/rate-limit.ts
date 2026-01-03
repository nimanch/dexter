function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function parseEnvInt(name: string): number | undefined {
  const raw = (process.env[name] ?? '').trim();
  if (!raw) return undefined;
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) && n >= 0 ? n : undefined;
}

export interface LlmRateLimitConfig {
  /** Max number of in-flight LLM requests. */
  maxConcurrency: number;

  /** Optional fixed delay between request starts. */
  minIntervalMs: number;

  /** Optional request pacing (requests per minute). */
  requestsPerMinute?: number;

  /** Optional token pacing (estimated tokens per minute). */
  tokensPerMinute?: number;

  /** Optional random jitter added to each scheduled start. */
  jitterMs: number;
}

export function getLlmRateLimitConfigFromEnv(): LlmRateLimitConfig {
  return {
    maxConcurrency: parseEnvInt('LLM_MAX_CONCURRENCY') ?? 1,
    minIntervalMs: parseEnvInt('LLM_MIN_INTERVAL_MS') ?? 0,
    requestsPerMinute: parseEnvInt('LLM_REQUESTS_PER_MINUTE'),
    tokensPerMinute: parseEnvInt('LLM_TOKENS_PER_MINUTE'),
    jitterMs: parseEnvInt('LLM_RATE_LIMIT_JITTER_MS') ?? 250,
  };
}

class Semaphore {
  private available: number;
  private readonly waiters: Array<() => void> = [];

  constructor(count: number) {
    this.available = Math.max(1, count);
  }

  async acquire(): Promise<void> {
    if (this.available > 0) {
      this.available -= 1;
      return;
    }

    await new Promise<void>((resolve) => {
      this.waiters.push(resolve);
    });
  }

  release(): void {
    const next = this.waiters.shift();
    if (next) {
      next();
      return;
    }
    this.available += 1;
  }
}

/**
 * Simple global limiter to reduce 429s.
 *
 * It supports:
 * - in-flight concurrency cap
 * - minimum start interval
 * - RPM pacing
 * - (estimated) TPM pacing
 */
export class LlmRateLimiter {
  private readonly sem: Semaphore;
  private readonly cfg: LlmRateLimitConfig;

  // Pacing is serialized so concurrent callers don't race cursors.
  private pacing: Promise<void> = Promise.resolve();

  private nextStartAtMs = 0;
  private nextRequestAtMs = 0;
  private nextTokenAtMs = 0;

  constructor(cfg: LlmRateLimitConfig) {
    this.cfg = cfg;
    this.sem = new Semaphore(cfg.maxConcurrency);
  }

  /**
   * Acquires a permit (concurrency + pacing) and returns a release function.
   * Useful for streaming calls where you need to hold the permit for the
   * duration of the stream while yielding results.
   */
  async acquirePermit(estimatedTokens: number): Promise<() => void> {
    await this.sem.acquire();
    try {
      await this.pace(estimatedTokens);
      return () => this.sem.release();
    } catch (e) {
      this.sem.release();
      throw e;
    }
  }

  async run<T>(estimatedTokens: number, fn: () => Promise<T>): Promise<T> {
    const release = await this.acquirePermit(estimatedTokens);
    try {
      return await fn();
    } finally {
      release();
    }
  }

  private pace(estimatedTokens: number): Promise<void> {
    const cfg = this.cfg;

    this.pacing = this.pacing.then(async () => {
      const now = Date.now();

      const rpm = cfg.requestsPerMinute && cfg.requestsPerMinute > 0 ? cfg.requestsPerMinute : undefined;
      const tpm = cfg.tokensPerMinute && cfg.tokensPerMinute > 0 ? cfg.tokensPerMinute : undefined;

      const requestDelay = rpm ? Math.ceil(60000 / rpm) : 0;
      const msPerToken = tpm ? 60000 / tpm : 0;
      const tokenCost = tpm ? Math.max(0, estimatedTokens) : 0;
      const tokenDelay = tpm ? Math.ceil(tokenCost * msPerToken) : 0;

      const jitter = cfg.jitterMs > 0 ? Math.floor(Math.random() * cfg.jitterMs) : 0;

      const startAt = Math.max(
        now,
        this.nextStartAtMs,
        rpm ? this.nextRequestAtMs : 0,
        tpm ? this.nextTokenAtMs : 0
      );

      // Reserve the slot.
      this.nextStartAtMs = startAt + cfg.minIntervalMs + jitter;
      if (rpm) this.nextRequestAtMs = startAt + requestDelay;
      if (tpm) this.nextTokenAtMs = startAt + tokenDelay;

      const waitMs = startAt - now;
      if (waitMs > 0) {
        await sleep(waitMs);
      }
    });

    return this.pacing;
  }
}

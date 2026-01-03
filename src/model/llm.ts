import { AzureChatOpenAI, ChatOpenAI } from '@langchain/openai';
import { ChatAnthropic } from '@langchain/anthropic';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { StructuredToolInterface } from '@langchain/core/tools';
import { Runnable } from '@langchain/core/runnables';
import { z } from 'zod';
import { DEFAULT_SYSTEM_PROMPT } from '../agent/prompts.js';
import { getLlmRateLimitConfigFromEnv, LlmRateLimiter } from './rate-limit.js';

export const DEFAULT_MODEL = 'gpt-5.2';

function disableLangSmithTracingIfMisconfigured(): void {
  const tracingFlag = (process.env.LANGSMITH_TRACING ?? process.env.LANGCHAIN_TRACING_V2 ?? '').trim();
  const tracingEnabled = tracingFlag.toLowerCase() === 'true';
  if (!tracingEnabled) return;

  const apiKey = (process.env.LANGSMITH_API_KEY ?? '').trim();
  const placeholder = apiKey === '' || apiKey === 'your-api-key' || apiKey.startsWith('your-api-key');
  if (!placeholder) return;

  // Prevent noisy 403s like "Failed to send multipart request" from LangSmith.
  process.env.LANGSMITH_TRACING = 'false';
  process.env.LANGCHAIN_TRACING_V2 = 'false';
}

// Run once on module import so even small one-off scripts (bun -e ...) behave.
disableLangSmithTracingIfMisconfigured();

function parseEnvInt(name: string): number | undefined {
  const raw = (process.env[name] ?? '').trim();
  if (!raw) return undefined;
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) && n >= 0 ? n : undefined;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function isRateLimitError(err: unknown): boolean {
  if (!err) return false;

  const anyErr = err as Record<string, unknown>;
  const message = String((anyErr.message ?? '') as string).toLowerCase();
  const status = anyErr.status ?? (anyErr.response as Record<string, unknown> | undefined)?.status;

  if (status === 429) return true;
  if (message.includes('429')) return true;
  if (message.includes('rate limit')) return true;
  if (message.includes('too many requests')) return true;

  return false;
}

function tryGetRetryAfterMs(err: unknown): number | undefined {
  if (!err) return undefined;
  const anyErr = err as Record<string, unknown>;

  const tryParseRetryAfter = (value: unknown): number | undefined => {
    if (value == null) return undefined;
    const raw = String(value).trim();
    if (!raw) return undefined;
    const seconds = Number(raw);
    if (Number.isFinite(seconds) && seconds >= 0) {
      return Math.ceil(seconds * 1000);
    }
    return undefined;
  };

  // Common shapes: err.response.headers.get('retry-after') / err.headers['retry-after']
  const response = anyErr.response as Record<string, unknown> | undefined;
  const headers = (response?.headers ?? anyErr.headers ?? (anyErr.cause as Record<string, unknown> | undefined)?.headers) as
    | Record<string, unknown>
    | undefined;

  const getHeader = (key: string): unknown => {
    if (!headers) return undefined;

    // Fetch Headers-like
    const maybeGet = headers as unknown as { get?: (k: string) => string | null };
    if (typeof maybeGet.get === 'function') {
      return maybeGet.get(key);
    }

    // Plain object
    const lowerKey = key.toLowerCase();
    for (const [k, v] of Object.entries(headers)) {
      if (k.toLowerCase() === lowerKey) return v;
    }
    return undefined;
  };

  const headerRetryAfter = getHeader('retry-after');
  const parsed = tryParseRetryAfter(headerRetryAfter);
  if (parsed != null) return parsed;

  // Best-effort parse from message
  const message = String((anyErr.message ?? '') as string);
  const match = message.match(/retry\s*after\s*(\d+(?:\.\d+)?)\s*(s|sec|secs|second|seconds)?/i);
  if (match) {
    const n = Number(match[1]);
    if (Number.isFinite(n) && n >= 0) return Math.ceil(n * 1000);
  }

  return undefined;
}

function getRetrySettingsFromEnv(): {
  maxAttempts: number;
  baseDelayMs: number;
  maxDelayMs: number;
  jitterRatio: number;
} {
  const maxAttempts = parseEnvInt('LLM_RETRY_MAX_ATTEMPTS') ?? 6;
  const baseDelayMs = parseEnvInt('LLM_RETRY_BASE_DELAY_MS') ?? 750;
  const maxDelayMs = parseEnvInt('LLM_RETRY_MAX_DELAY_MS') ?? 30000;
  const jitterRatioRaw = (process.env.LLM_RETRY_JITTER_RATIO ?? '').trim();
  const jitterRatio = jitterRatioRaw ? Number(jitterRatioRaw) : 0.2;

  return {
    maxAttempts: Math.max(1, maxAttempts),
    baseDelayMs: Math.max(0, baseDelayMs),
    maxDelayMs: Math.max(0, maxDelayMs),
    jitterRatio: Number.isFinite(jitterRatio) && jitterRatio >= 0 ? jitterRatio : 0.2,
  };
}

function withJitter(ms: number, ratio: number): number {
  if (ms <= 0) return 0;
  const jitter = Math.floor(ms * ratio * Math.random());
  return ms + jitter;
}

async function withRetry<T>(fn: () => Promise<T>): Promise<T> {
  const { maxAttempts, baseDelayMs, maxDelayMs, jitterRatio } = getRetrySettingsFromEnv();

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      return await fn();
    } catch (e) {
      if (attempt === maxAttempts - 1) throw e;

      const retryAfterMs = tryGetRetryAfterMs(e);
      const is429 = isRateLimitError(e);

      const backoff = Math.min(maxDelayMs, baseDelayMs * 2 ** attempt);
      const delayMs = retryAfterMs ?? (is429 ? Math.max(backoff, 1000) : backoff);
      await sleep(withJitter(delayMs, jitterRatio));
    }
  }

  throw new Error('Unreachable');
}

function estimateTokensFromText(text: string): number {
  // Heuristic: ~4 chars per token in English.
  const chars = text.length;
  return Math.max(1, Math.ceil(chars / 4));
}

const globalRateLimiter = new LlmRateLimiter(getLlmRateLimitConfigFromEnv());

// Model provider configuration
interface ModelOpts {
  streaming: boolean;
}

type ModelFactory = (name: string, opts: ModelOpts) => BaseChatModel;

function getApiKey(envVar: string, providerName: string): string {
  const apiKey = process.env[envVar];
  if (!apiKey) {
    throw new Error(`${envVar} not found in environment variables`);
  }
  return apiKey;
}

function getEnvVar(envVar: string): string | undefined {
  const value = process.env[envVar];
  const trimmed = value?.trim();
  return trimmed ? trimmed : undefined;
}

const MODEL_PROVIDERS: Record<string, ModelFactory> = {
  'azure-': (name, opts) => {
    const deploymentName =
      getEnvVar('AZURE_OPENAI_API_DEPLOYMENT_NAME') ?? name.slice('azure-'.length);

    if (!deploymentName) {
      throw new Error(
        'Azure model name must be in the form azure-<deploymentName> or AZURE_OPENAI_API_DEPLOYMENT_NAME must be set.'
      );
    }

    const azureOpenAIBasePath = getEnvVar('AZURE_OPENAI_BASE_PATH');

    if (!azureOpenAIBasePath) {
      throw new Error(
        'Azure OpenAI configuration missing. Set AZURE_OPENAI_BASE_PATH (ending with /openai/deployments/).'
      );
    }

    return new AzureChatOpenAI({
      ...opts,
      // Azure uses a deployment; we also set `model` for compatibility.
      model: deploymentName,
      azureOpenAIApiKey: getApiKey('AZURE_OPENAI_API_KEY', 'Azure OpenAI'),
      azureOpenAIApiDeploymentName: deploymentName,
      azureOpenAIApiVersion: getEnvVar('AZURE_OPENAI_API_VERSION'),
      azureOpenAIBasePath,
    });
  },
  'claude-': (name, opts) =>
    new ChatAnthropic({
      model: name,
      ...opts,
      apiKey: getApiKey('ANTHROPIC_API_KEY', 'Anthropic'),
    }),
  'gemini-': (name, opts) =>
    new ChatGoogleGenerativeAI({
      model: name,
      ...opts,
      apiKey: getApiKey('GOOGLE_API_KEY', 'Google'),
    }),
};

const DEFAULT_PROVIDER: ModelFactory = (name, opts) =>
  new ChatOpenAI({
    model: name,
    ...opts,
    apiKey: process.env.OPENAI_API_KEY,
  });

export function getChatModel(
  modelName: string = DEFAULT_MODEL,
  streaming: boolean = false
): BaseChatModel {
  const opts: ModelOpts = { streaming };
  const prefix = Object.keys(MODEL_PROVIDERS).find((p) => modelName.startsWith(p));
  const factory = prefix ? MODEL_PROVIDERS[prefix] : DEFAULT_PROVIDER;
  return factory(modelName, opts);
}

interface CallLlmOptions {
  model?: string;
  systemPrompt?: string;
  outputSchema?: z.ZodType<unknown>;
  tools?: StructuredToolInterface[];
}

export async function callLlm(prompt: string, options: CallLlmOptions = {}): Promise<unknown> {
  const { model = DEFAULT_MODEL, systemPrompt, outputSchema, tools } = options;
  const finalSystemPrompt = systemPrompt || DEFAULT_SYSTEM_PROMPT;

  const promptTemplate = ChatPromptTemplate.fromMessages([
    ['system', finalSystemPrompt],
    ['user', '{prompt}'],
  ]);

  const llm = getChatModel(model, false);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  let runnable: Runnable<any, any> = llm;

  if (outputSchema) {
    runnable = llm.withStructuredOutput(outputSchema);
  } else if (tools && tools.length > 0 && llm.bindTools) {
    runnable = llm.bindTools(tools);
  }

  const chain = promptTemplate.pipe(runnable);

  const estimatedTokens =
    estimateTokensFromText(prompt) +
    estimateTokensFromText(finalSystemPrompt) +
    (tools && tools.length > 0 ? 250 : 0);

  const result = await globalRateLimiter.run(estimatedTokens, () =>
    withRetry(() => chain.invoke({ prompt }))
  );

  // If no outputSchema and no tools, extract content from AIMessage
  // When tools are provided, return the full AIMessage to preserve tool_calls
  if (!outputSchema && !tools && result && typeof result === 'object' && 'content' in result) {
    return (result as { content: string }).content;
  }
  return result;
}

export async function* callLlmStream(
  prompt: string,
  options: { model?: string; systemPrompt?: string } = {}
): AsyncGenerator<string> {
  const { model = DEFAULT_MODEL, systemPrompt } = options;
  const finalSystemPrompt = systemPrompt || DEFAULT_SYSTEM_PROMPT;

  const promptTemplate = ChatPromptTemplate.fromMessages([
    ['system', finalSystemPrompt],
    ['user', '{prompt}'],
  ]);

  const llm = getChatModel(model, true);
  const chain = promptTemplate.pipe(llm);

  const estimatedTokens = estimateTokensFromText(prompt) + estimateTokensFromText(finalSystemPrompt);

  // For streaming, retry only if the connection fails before yielding any tokens.
  const { maxAttempts } = getRetrySettingsFromEnv();

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    let receivedAny = false;

    try {
      const release = await globalRateLimiter.acquirePermit(estimatedTokens);
      try {
        const stream = await chain.stream({ prompt });

        for await (const chunk of stream) {
          if (chunk && typeof chunk === 'object' && 'content' in chunk) {
            const content = (chunk as { content?: unknown }).content;
            if (typeof content === 'string' && content) {
              receivedAny = true;
              yield content;
            }
          }
        }
      } finally {
        release();
      }

      return;
    } catch (e) {
      // Don't retry mid-stream.
      if (receivedAny) throw e;
      if (attempt === maxAttempts - 1) throw e;

      // Use the same 429-aware backoff.
      const retryAfterMs = tryGetRetryAfterMs(e);
      const is429 = isRateLimitError(e);
      const { baseDelayMs, maxDelayMs, jitterRatio } = getRetrySettingsFromEnv();
      const backoff = Math.min(maxDelayMs, baseDelayMs * 2 ** attempt);
      const delayMs = retryAfterMs ?? (is429 ? Math.max(backoff, 1000) : backoff);
      await sleep(withJitter(delayMs, jitterRatio));
    }
  }
}

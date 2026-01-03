import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { callApi } from './api.js';
import { callFmp } from './fmp-api.js';
import { runWithFinanceProviderFallback, unsupportedByProvider } from './provider.js';
import { formatToolResult } from '../types.js';

const SegmentedRevenuesInputSchema = z.object({
  ticker: z
    .string()
    .describe(
      "The stock ticker symbol to fetch segmented revenues for. For example, 'AAPL' for Apple."
    ),
  period: z
    .enum(['annual', 'quarterly'])
    .describe(
      "The reporting period for the segmented revenues. 'annual' for yearly, 'quarterly' for quarterly."
    ),
  limit: z.number().default(10).describe('The number of past periods to retrieve.'),
});

export const getSegmentedRevenues = new DynamicStructuredTool({
  name: 'get_segmented_revenues',
  description: `Provides a detailed breakdown of a company's revenue by operating segments, such as products, services, or geographic regions. Useful for analyzing the composition of a company's revenue.`,
  schema: SegmentedRevenuesInputSchema,
  func: async (input) => {
    return runWithFinanceProviderFallback('get_segmented_revenues', async (provider) => {
      if (provider === 'alphavantage') {
        throw unsupportedByProvider(provider, 'get_segmented_revenues');
      }

      if (provider === 'fmp') {
        const fmpPeriod = input.period === 'quarterly' ? 'quarter' : 'annual';
        const symbol = input.ticker.toUpperCase();

        const [product, geographic] = await Promise.all([
          callFmp('/revenue-product-segmentation', { symbol, period: fmpPeriod, structure: 'flat' }),
          // Note: docs page may differ; endpoint is expected to exist.
          callFmp('/revenue-geographic-segmentation', { symbol, period: fmpPeriod, structure: 'flat' }),
        ]);

        const slice = (d: Record<string, unknown>): unknown => {
          if (Array.isArray(d)) return d.slice(0, input.limit);
          return d;
        };

        return formatToolResult(
          {
            product: slice(product.data),
            geographic: slice(geographic.data),
          },
          [product.url, geographic.url]
        );
      }

      const params = {
        ticker: input.ticker,
        period: input.period,
        limit: input.limit,
      };
      const { data, url } = await callApi('/financials/segmented-revenues/', params);
      return formatToolResult(data.segmented_revenues || {}, [url]);
    });
  },
});


import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { callApi } from './api.js';
import { callFmp } from './fmp-api.js';
import { runWithFinanceProviderFallback, unsupportedByProvider } from './provider.js';
import { formatToolResult } from '../types.js';

const AnalystEstimatesInputSchema = z.object({
  ticker: z
    .string()
    .describe(
      "The stock ticker symbol to fetch analyst estimates for. For example, 'AAPL' for Apple."
    ),
  period: z
    .enum(['annual', 'quarterly'])
    .default('annual')
    .describe("The period for the estimates, either 'annual' or 'quarterly'."),
});

export const getAnalystEstimates = new DynamicStructuredTool({
  name: 'get_analyst_estimates',
  description: `Retrieves analyst estimates for a given company ticker, including metrics like estimated EPS. Useful for understanding consensus expectations, assessing future growth prospects, and performing valuation analysis.`,
  schema: AnalystEstimatesInputSchema,
  func: async (input) => {
    return runWithFinanceProviderFallback('get_analyst_estimates', async (provider) => {
      if (provider === 'alphavantage') {
        throw unsupportedByProvider(provider, 'get_analyst_estimates');
      }

      if (provider === 'fmp') {
        const fmpPeriod = input.period === 'quarterly' ? 'quarter' : 'annual';
        const { data, url } = await callFmp('/analyst-estimates', {
          symbol: input.ticker.toUpperCase(),
          period: fmpPeriod,
          page: 0,
          limit: 50,
        });
        return formatToolResult(data, [url]);
      }

      const params = {
        ticker: input.ticker,
        period: input.period,
      };
      const { data, url } = await callApi('/analyst-estimates/', params);
      return formatToolResult(data.analyst_estimates || [], [url]);
    });
  },
});


import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { callApi } from './api.js';
import { callAlphaVantage } from './alphavantage-api.js';
import { callFmp } from './fmp-api.js';
import { runWithFinanceProviderFallback, unsupportedByProvider } from './provider.js';
import { formatToolResult } from '../types.js';

const FinancialStatementsInputSchema = z.object({
  ticker: z
    .string()
    .describe(
      "The stock ticker symbol to fetch financial statements for. For example, 'AAPL' for Apple."
    ),
  period: z
    .enum(['annual', 'quarterly', 'ttm'])
    .describe(
      "The reporting period for the financial statements. 'annual' for yearly, 'quarterly' for quarterly, and 'ttm' for trailing twelve months."
    ),
  limit: z
    .number()
    .default(10)
    .describe(
      'Maximum number of report periods to return (default: 10). Returns the most recent N periods based on the period type.'
    ),
  report_period_gt: z
    .string()
    .optional()
    .describe('Filter for financial statements with report periods after this date (YYYY-MM-DD).'),
  report_period_gte: z
    .string()
    .optional()
    .describe(
      'Filter for financial statements with report periods on or after this date (YYYY-MM-DD).'
    ),
  report_period_lt: z
    .string()
    .optional()
    .describe('Filter for financial statements with report periods before this date (YYYY-MM-DD).'),
  report_period_lte: z
    .string()
    .optional()
    .describe(
      'Filter for financial statements with report periods on or before this date (YYYY-MM-DD).'
    ),
});

function createParams(input: z.infer<typeof FinancialStatementsInputSchema>): Record<string, string | number | undefined> {
  return {
    ticker: input.ticker,
    period: input.period,
    limit: input.limit,
    report_period_gt: input.report_period_gt,
    report_period_gte: input.report_period_gte,
    report_period_lt: input.report_period_lt,
    report_period_lte: input.report_period_lte,
  };
}

export const getIncomeStatements = new DynamicStructuredTool({
  name: 'get_income_statements',
  description: `Fetches a company's income statements, detailing its revenues, expenses, net income, etc. over a reporting period. Useful for evaluating a company's profitability and operational efficiency.`,
  schema: FinancialStatementsInputSchema,
  func: async (input) => {
    const params = createParams(input);
    return runWithFinanceProviderFallback('get_income_statements', async (provider) => {
      if (provider === 'alphavantage') {
        const { data, url } = await callAlphaVantage('INCOME_STATEMENT', {
          symbol: input.ticker.toUpperCase(),
        });
        return formatToolResult(data, [url]);
      }

      if (provider === 'fmp') {
        if (input.period === 'ttm') {
          throw unsupportedByProvider(provider, 'get_income_statements (TTM not implemented for FMP)');
        }

        const fmpPeriod = input.period === 'quarterly' ? 'quarter' : 'annual';
        const { data, url } = await callFmp('/income-statement', {
          symbol: input.ticker.toUpperCase(),
          period: fmpPeriod,
          limit: input.limit,
        });
        return formatToolResult(data, [url]);
      }

      const { data, url } = await callApi('/financials/income-statements/', params);
      return formatToolResult(data.income_statements || {}, [url]);
    });
  },
});

export const getBalanceSheets = new DynamicStructuredTool({
  name: 'get_balance_sheets',
  description: `Retrieves a company's balance sheets, providing a snapshot of its assets, liabilities, shareholders' equity, etc. at a specific point in time. Useful for assessing a company's financial position.`,
  schema: FinancialStatementsInputSchema,
  func: async (input) => {
    const params = createParams(input);
    return runWithFinanceProviderFallback('get_balance_sheets', async (provider) => {
      if (provider === 'alphavantage') {
        const { data, url } = await callAlphaVantage('BALANCE_SHEET', {
          symbol: input.ticker.toUpperCase(),
        });
        return formatToolResult(data, [url]);
      }

      if (provider === 'fmp') {
        if (input.period === 'ttm') {
          throw unsupportedByProvider(provider, 'get_balance_sheets (TTM not implemented for FMP)');
        }

        const fmpPeriod = input.period === 'quarterly' ? 'quarter' : 'annual';
        const { data, url } = await callFmp('/balance-sheet-statement', {
          symbol: input.ticker.toUpperCase(),
          period: fmpPeriod,
          limit: input.limit,
        });
        return formatToolResult(data, [url]);
      }

      const { data, url } = await callApi('/financials/balance-sheets/', params);
      return formatToolResult(data.balance_sheets || {}, [url]);
    });
  },
});

export const getCashFlowStatements = new DynamicStructuredTool({
  name: 'get_cash_flow_statements',
  description: `Retrieves a company's cash flow statements, showing how cash is generated and used across operating, investing, and financing activities. Useful for understanding a company's liquidity and solvency.`,
  schema: FinancialStatementsInputSchema,
  func: async (input) => {
    const params = createParams(input);
    return runWithFinanceProviderFallback('get_cash_flow_statements', async (provider) => {
      if (provider === 'alphavantage') {
        const { data, url } = await callAlphaVantage('CASH_FLOW', {
          symbol: input.ticker.toUpperCase(),
        });
        return formatToolResult(data, [url]);
      }

      if (provider === 'fmp') {
        if (input.period === 'ttm') {
          throw unsupportedByProvider(provider, 'get_cash_flow_statements (TTM not implemented for FMP)');
        }

        const fmpPeriod = input.period === 'quarterly' ? 'quarter' : 'annual';
        const { data, url } = await callFmp('/cash-flow-statement', {
          symbol: input.ticker.toUpperCase(),
          period: fmpPeriod,
          limit: input.limit,
        });
        return formatToolResult(data, [url]);
      }

      const { data, url } = await callApi('/financials/cash-flow-statements/', params);
      return formatToolResult(data.cash_flow_statements || {}, [url]);
    });
  },
});

export const getAllFinancialStatements = new DynamicStructuredTool({
  name: 'get_all_financial_statements',
  description: `Retrieves all three financial statements (income statements, balance sheets, and cash flow statements) for a company in a single API call. This is more efficient than calling each statement type separately when you need all three for comprehensive financial analysis.`,
  schema: FinancialStatementsInputSchema,
  func: async (input) => {
    const params = createParams(input);
    return runWithFinanceProviderFallback('get_all_financial_statements', async (provider) => {
      if (provider === 'alphavantage') {
        const symbol = input.ticker.toUpperCase();
        const [income, balance, cash] = await Promise.all([
          callAlphaVantage('INCOME_STATEMENT', { symbol }),
          callAlphaVantage('BALANCE_SHEET', { symbol }),
          callAlphaVantage('CASH_FLOW', { symbol }),
        ]);

        return formatToolResult(
          {
            income_statement: income.data,
            balance_sheet: balance.data,
            cash_flow: cash.data,
          },
          [income.url, balance.url, cash.url]
        );
      }

      if (provider === 'fmp') {
        if (input.period === 'ttm') {
          throw unsupportedByProvider(provider, 'get_all_financial_statements (TTM not implemented for FMP)');
        }

        const symbol = input.ticker.toUpperCase();
        const fmpPeriod = input.period === 'quarterly' ? 'quarter' : 'annual';
        const [income, balance, cash] = await Promise.all([
          callFmp('/income-statement', { symbol, period: fmpPeriod, limit: input.limit }),
          callFmp('/balance-sheet-statement', { symbol, period: fmpPeriod, limit: input.limit }),
          callFmp('/cash-flow-statement', { symbol, period: fmpPeriod, limit: input.limit }),
        ]);

        return formatToolResult(
          {
            income_statement: income.data,
            balance_sheet: balance.data,
            cash_flow: cash.data,
          },
          [income.url, balance.url, cash.url]
        );
      }

      const { data, url } = await callApi('/financials/', params);
      return formatToolResult(data.financials || {}, [url]);
    });
  },
});


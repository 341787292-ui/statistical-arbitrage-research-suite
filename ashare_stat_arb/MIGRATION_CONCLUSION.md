# A-Share Method Migration Conclusion

## Answer

The tested public-data approximation does not support migrating the paper's
residual methodology into the declared cash-only CSI 500 index-enhancement
workflow.

This conclusion applies to the tested route, not to every possible residual
strategy:

- monthly five-factor PCA residual construction;
- five-day cross-sectional residual reversal rank;
- 100-name point-in-time BaoStock pilot;
- equal-weight pilot benchmark;
- long-only benchmark-relative optimization;
- declared A-share next-open execution rules and costs;
- three frozen mappings: daily continuous rank, five-session low-turnover, and
  daily buffered state.

## Reasoning chain

1. The paper-aligned OU mechanism was economically negligible in residual
   space and was not rescued by an alternative residual-history definition.
2. A model-free audit found a narrower one- and five-day cross-sectional rank
   pattern, so one fixed five-day signal was allowed to proceed.
3. Daily mapping retained 2.29% gross excess but incurred 9.48% annualized cost
   drag.
4. Five-session mapping reduced costs to 1.07% but reduced gross excess to
   0.30%.
5. A daily state buffer produced 0.59% gross excess and 3.70% cost drag.
6. Every executable mapping had negative net excess and failed robustness
   gates. The holdout was therefore never opened.

## What has been learned

The main obstacle is not one incorrect sign or one expensive commission
assumption. The observed residual-rank information is both weak and short-
lived. Capturing it quickly requires too much trading, while reducing trading
causes most of the gross effect to disappear.

## What has not been proved

The study does not prove that:

- all A-share residual information is useless;
- the full paper cannot work with richer data and a different model;
- an institutional factor library would not improve residual quality;
- every CSI 500 index-enhancement strategy fails;
- official historical weights would produce identical numbers.

## Practical recommendation

Do not buy expensive data or open the 2023-2025 holdout to continue this exact
branch. The free pilot has already rejected the tested mechanism-to-portfolio
path. Any future project should begin with a materially new research question
and a new untouched evaluation period, rather than another threshold change.

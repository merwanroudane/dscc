"""60-second DSCC example: two Fama-French factors, last 1,000 trading days of the paper's sample."""
import dscc

ff = dscc.datasets.load_ff5_daily(end="2022-03-31")          # downloads once, then cached
data = ff[["Mkt-RF", "HML"]].iloc[-1000:]

res = dscc.DSCC(data, dynamics="egarch").fit(n_starts=3, seed=1)
print(res.summary())

fc = res.forecast(H=21, init="average")                    # one trading month ahead
print("\nForecast correlation, average over the next 21 days:",
      round(dscc.correlation_from_cov(fc.mean_sigma)[0, 1], 3))
print("Minimum-variance weights:", dscc.gmv_weights(fc.mean_sigma).round(3))

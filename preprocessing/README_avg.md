## preprocessing

The temporal average is performed in three steps:

- computation of daily means with `preprocessing/daily_mean/daily_mean.sh`
- computation of monthly means with `preprocessing/average_daily_mean/average_daily_means.sh`
- computation of the global mean with `preprocessing/final_mean/final_mean.sh`
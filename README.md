# Internal Tide analysis in NEMO run eNATL60

## Organisation of the repository
* docs: working documents and notes
* code: python packages / scripts developed for the analysis
* training: folder for trials et al (sandbox for root directory)
* dev_notebooks: notebook used for developping some of the routines (sandbox for code/)
* tutos: notebooks illustrating the use of the various analysis code and routines written
* itidenatl: core library containing useful objects/methods

Some processing notebooks (in clean form) can be at the root of the repository.

### preprocessing

The temporal average is performed in three steps:

- computation of daily means with `preprocessing/daily_mean/daily_mean.sh`
- computation of monthly means with `preprocessing/average_daily_mean/average_daily_means.sh`
- computation of the global mean with `preprocessing/final_mean/final_mean.sh`

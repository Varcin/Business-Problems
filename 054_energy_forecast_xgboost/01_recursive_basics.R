# AUTOREGRESSIVE FORECASTING ----
# RECURSIVE BASICS ----
# **** ----

# LIBRARIES ----

# Use Development version of modeltime:
#remotes::install_github("business-science/modeltime")

library(modeltime)
library(tidymodels)
library(tidyverse)
library(lubridate)
library(timetk) # has example datasets.

# MULTIPLE TIME SERIES (PANEL DATA) -----

m4_monthly # coming from the timetk package.

m4_monthly %>%
    group_by(id) %>%
    plot_time_series(date, value)

FORECAST_HORIZON <- 24

# Extend the dataframe for 24 months which is our forecast horizon.
# future_frame is a function of timetk package.
m4_extended <- m4_monthly %>%
    group_by(id) %>%
    future_frame(
        .length_out = FORECAST_HORIZON,
        .bind_data  = TRUE
    ) %>%
    ungroup()

# TRANSFORM FUNCTION ----
# - NOTE - We create lags by group
lag_transformer_grouped <- function(data){
    data %>%
        group_by(id) %>%
        tk_augment_lags(value, .lags = 1:FORECAST_HORIZON) %>%
        ungroup()
}

m4_lags <- m4_extended %>%
    lag_transformer_grouped()
# lag transformer creates 24 lag columns. 1st column is lagged only once. 2nd column lagged twice. Third column is lagged three times and it goes like that until 24.

# Dropping NAs removes the first and last 24 records. last 24 records are coming from the ones that we included with the future_frame function.
train_data <- m4_lags %>%
    drop_na()

future_data <- m4_lags %>%
    filter(is.na(value))

# Modeling Autoregressive Panel Data ----

# * Normal Linear Regression ----
model_fit_lm <- linear_reg() %>%
    set_engine("lm") %>% #parsnip package
    fit(value ~ ., data = train_data)

# * Recursive Linear Regression ----
model_fit_lm_recursive <- model_fit_lm %>%
    # modeltime function
    recursive(

        # We add an id = "id" to specify the groups
        id         = "id",

        # Supply the transformation function used to generate the lags
        transform  = lag_transformer_grouped,

        # We use panel_tail() to grab tail by groups
        # panel tail grabs the last 24 observations. It creates a df.
        train_tail = panel_tail(train_data, id, FORECAST_HORIZON)
    )

# MODELTIME WORKFLOW FOR PANEL DATA ----
# Until group_by it does forecasting and the rest just plots the results.
modeltime_table(
    model_fit_lm,
    model_fit_lm_recursive
) %>%
    modeltime_forecast(
        new_data    = future_data,
        actual_data = m4_monthly,
        keep_data   = TRUE
    ) %>%
    group_by(id) %>%
    plot_modeltime_forecast(
        .interactive = TRUE,
        .conf_interval_show = FALSE
    )

# Notice LM gives us an error while the recursive makes the forecast.
# This happens because recursive() tells the NA values to be filled in use the lag transformer function.

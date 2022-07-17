# ANOMALY DETECTION ----
# METHOD: H2O ISOLATION FOREST ----
# CASE: DETECTING CREDIT CARD FRAUD ----
# Kaggle Data: https://www.kaggle.com/mlg-ulb/creditcardfraud

# 1.0 LIBRARIES ----
library(h2o) # machine learning models
library(tidyverse)
library(tidyquant)
library(yardstick) # Use devtools::install_github("tidymodels/yardstick")
library(vroom) # fast data load
library(plotly) # interactive plots


credit_card_tbl <- vroom("data/creditcard.csv")


# 1.1 CLASS IMBALANCE ----
credit_card_tbl %>%
    count(Class) %>%
    mutate(prop = n / sum(n))
# very imbalanced dataset

# 1.2 AMOUNT SPENT VS FRAUD ----
g <- credit_card_tbl %>%
    select(Amount, Class) %>%
    ggplot(aes(Amount, fill = as.factor(Class))) +
    # geom_histogram() +
    geom_density(alpha = 0.3) +
    facet_wrap(~ Class, scales = "free_y", ncol = 1) +
    scale_x_log10(label = scales::dollar_format()) + # apply log scale. data is very skewed
    theme_minimal() +
    labs(title = "Fraud by Amount Spent", 
         fill = "Fraud")

ggplotly(g)

# 2.0 H2O ISOLATION FOREST ----
h2o.init()

credit_card_h2o <- as.h2o(credit_card_tbl)
credit_card_h2o

# Class is the column we are trying to predict
target <- "Class"
# Isolation forest doesn't need labels. It is not a supervised learning model.
predictors <- setdiff(names(credit_card_h2o), target)

isoforest <- h2o.isolationForest(
    training_frame = credit_card_h2o, # isolation forest doesn't need labels. Creating labeled data is time intensive.
    x      = predictors,
    ntrees = 100, # 100 is fine for isolation forest algorithm
    seed   = 1234
)

isoforest

# 3.0 PREDICTIONS ----
predictions <- predict(isoforest, newdata = credit_card_h2o)
predictions
# there are two columns. Predict and mean length.
# predict is basically the likelihood that an observation is an outlier which is based on mean length. Mean length is the number of decision trees used to isolate that observation. 



# 4.0 METRICS ----

# 4.1 Quantile ----
h2o.hist(predictions[,"predict"])
h2o.hist(predictions[,"mean_length"])

quantile <- h2o.quantile(predictions, probs = 0.99)
quantile

thresh <- quantile["predictQuantiles"]

predictions$outlier <- predictions$predict > thresh %>% as.numeric()
predictions$class <- credit_card_h2o$Class

predictions

predictions_tbl <- as_tibble(predictions) %>%
    mutate(class = as.factor(class)) %>%
    mutate(outlier = as.factor(outlier))
predictions_tbl



# 4.2 Confusion Matrix ----

predictions_tbl %>% conf_mat(class, outlier)


# 4.3 ROC Curve ----
auc <- predictions_tbl %>% 
    roc_auc(class, predict, event_level = "second") %>% # area under the curve
    pull(.estimate) %>%
    round(3)
# 0.948 highly predictive power

roc_plot <- predictions_tbl %>% 
    roc_curve(class, predict, event_level = "second") %>%
    ggplot(aes(x = 1 - specificity, y = sensitivity)) +
    geom_path(color = palette_light()[1], size = 2) +
    geom_abline(lty = 3, size = 1) +
    theme_minimal() +
    labs(title = str_glue("ROC AUC: {auc}"), 
         subtitle = "Using H2O Isolation Forest")
# anomalies are closely related with fraud. 
roc_plot
ggsave(roc_plot, filename = "img/fraud_roc_auc_plot.png")


# 4.4 Precision vs Recall AUC ----
predictions_tbl %>% pr_auc(class, predict)


# 5.0 CONCLUSIONS ----
# - Anomalies (Outliers) are more often than not Fraudulent Transactions
# - Isolation Forest does a good job at detecting anomalous behaviour



# 6.0 Stabilize Predictions with PURRR

# 6.1 Repeatable Prediction Function ----
iso_forest <- function(seed) {
    
    target <- "Class"
    predictors <- setdiff(names(credit_card_h2o), target)
    
    isoforest <- h2o.isolationForest(
        training_frame = credit_card_h2o,
        x      = predictors,
        ntrees = 100, 
        seed   = seed
    )
    
    predictions <- predict(isoforest, newdata = credit_card_h2o)
    
    quantile <- h2o.quantile(predictions, probs = 0.99)
    
    thresh <- quantile["predictQuantiles"]
    
    # predictions$outlier <- predictions$predict > thresh %>% as.numeric()
    # predictions$class <- credit_card_h2o$Class
    
    predictions_tbl <- as_tibble(predictions) %>%
        # mutate(class = as.factor(class)) %>%
        mutate(row = row_number())
    predictions_tbl
    
}

iso_forest(123)

# 6.2 MAP TO MULTIPLE SEEDS ----
multiple_predictions_tbl <- tibble(seed = c(158, 8546, 4593)) %>%
    mutate(predictions = map(seed, iso_forest))

multiple_predictions_tbl

# 6.3 CALCULATE AVERAGE PREDICTIONS ----
stabilized_predictions_tbl <- multiple_predictions_tbl %>% 
    unnest(predictions) %>%
    select(row, seed, predict) %>%
    
    # Calculate stabilized predictions
    group_by(row) %>%
    summarize(mean_predict = mean(predict)) %>%
    ungroup() %>%
    
    # Combine with original data & important columns
    bind_cols(
        credit_card_tbl
    ) %>% 
    select(row, mean_predict, Time, V12, V15, Amount, Class) %>%
    
    # Detect Outliers
    mutate(outlier = ifelse(mean_predict > quantile(mean_predict, probs = 0.99), 1, 0)) %>%
    mutate(Class = as.factor(Class))

# 6.4 MEASURE ----
stabilized_predictions_tbl %>% pr_auc(Class, mean_predict)

# 6.5 VISUALIZE ----
# - Not Run Due to Time

stabilized_predictions_tbl %>%
    ggplot(aes(V12, V15, color = as.factor(outlier))) +
    geom_point(alpha = 0.2) +
    theme_minimal() +
    labs(title = "Anomaly Detected?", color = "Is Outlier?")

stabilized_predictions_tbl %>%
    ggplot(aes(V12, V15, color = as.factor(outlier))) +
    geom_point(alpha = 0.2) +
    theme_minimal() +
    labs(title = "Fraud Present?", color = "Is Fraud?")


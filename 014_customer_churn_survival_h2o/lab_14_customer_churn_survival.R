# - Churn Survival Analysis
# - Dataset comes from IBM Telco Customer Churn

# 1.0 LIBRARIES ----
library(devtools)

# Modeling
library(survival)
#devtools::install_github("tidymodels/censored")
library(censored)
library(parsnip)
library(broom)

# Advanced ML
library(h2o)
library(lime)

# EDA
library(correlationfunnel)

# Core & Data Viz
library(tidyverse)
library(plotly)
library(tidyquant)

# Check H2O Version to match Model Version
source("scripts/check_h2o_version.R")


# 2.0 DATA ----
# - KEY POINTS: tenure = Time, Churn = Target, Everything Else = Possible Predictors

customer_churn_tbl <- read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

customer_churn_tbl %>% glimpse()


# 3.0 EXPLORATORY DATA ANALYSIS ----

customer_churn_tbl %>% binarize()

# Check NA's - purrr/dplyr Data Wrangling
customer_churn_tbl %>%
    map_df(~ sum(is.na(.))) %>%
    gather() %>%
    arrange(desc(value))

customer_churn_tbl %>%
    filter(is.na(TotalCharges)) %>%
    glimpse()

# Prep Data: Remove non-predictive ID & fix NA's
# Looks like total charges for the first month is entered NA for new customers 
customer_churn_prep_tbl <- customer_churn_tbl %>%
    select(-customerID) %>%
    mutate(TotalCharges = case_when(
        is.na(TotalCharges) ~ MonthlyCharges,
        TRUE ~ TotalCharges
    )) 

# Correlation Funnel 
corr_funel_1 <- customer_churn_prep_tbl %>%
    binarize() %>% # Creates 1s and 0s for each category. If categorical it creates dummy variables, if numeric it creates bins as dummy variables.
    correlate(Churn__Yes) %>% # I am interested in Churn. That's what I will predict. 
    plot_correlation_funnel(interactive = FALSE, alpha = 0.7) +
    theme_minimal() + 
    xlab("Feature") +
    ylab("Correlation") + 
    theme(text=element_text(size=12))
corr_funel_1
ggsave(corr_funel_1, filename = "img/corr_funel_1.png", width = 9.93, height = 4)


# 4.0 SURVIVAL ANALYSIS ----
# I have selected below variables by looking at the correlation funnel
# I am skipping tenure because I will use that as the time varient feature of the model
train_tbl <- customer_churn_prep_tbl %>%
    mutate(
        Churn_Yes                     = Churn == "Yes",
        OnlineSecurity_No             = OnlineSecurity == "No",
        TechSupport_No                = TechSupport == "No",
        InternetService_FiberOptic    = InternetService %>% str_detect("Fiber"),
        PaymentMethod_ElectronicCheck = PaymentMethod %>% str_detect("Electronic"),
        OnlineBackup_No               = OnlineBackup == "No",
        DeviceProtection_No           = DeviceProtection == "No"
    ) %>%
    select(
        tenure, Churn_Yes, Contract, OnlineSecurity_No, TechSupport_No, InternetService_FiberOptic,
        PaymentMethod_ElectronicCheck, OnlineSecurity_No, DeviceProtection_No
    ) 


# 4.1 Survival Tables (Kaplan-Meier Method) ---- (simple model)
# Churn will be based on tenure and we want to look at this by contract.
survfit_simple <- survfit(Surv(tenure, Churn_Yes) ~ strata(Contract), data = train_tbl)
survfit_simple

# Mortality Table
tidy(survfit_simple)

# 4.2 Cox Regression Model (Multivariate) ----
# Cox Proportional Hazard
# select everything except Contract because that is going to be our stratification variable
# Use . to tell all the variables but then remove contract because it will be our stratification variable. 
model_coxph <- coxph(Surv(tenure, Churn_Yes) ~ . - Contract + strata(Contract), data = train_tbl)

# Overall performance
glance(model_coxph) %>% glimpse()

# Regression Estimates
tidy(model_coxph)

# Mortality Table
model_coxph %>%
    survfit() %>% # survfit gives the mortality table. 
    tidy()




# 5.0 SURVIVAL CURVES -----

plot_customer_survival <- function(object_survfit) {
    
    g <- tidy(object_survfit) %>%
        ggplot(aes(time, estimate, color = strata)) +
        geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.5, show.legend = F) +
        geom_line(size = 1) +
        theme_minimal() +
        scale_color_manual(values = unname(palette.colors(palette = "Tableau"))) +
        scale_y_continuous(labels = scales::percent_format()) +
        labs(title = "Churn Problem", color = "Contract Type", 
             x = "Days After Purchase", y = "Percentage of Customers Staying") +
        theme(legend.key=element_blank(),
              legend.position = "bottom") 
    g
    # ggplotly(g) %>% 
    #     plotly::layout(legend = list(orientation = "h", x = 0.2, y = -0.4))
}


plot_customer_survival(survfit_simple)
survfit_png <- plot_customer_survival(survfit_simple)

survfit_html <- plot_customer_survival(survfit_simple) %>% 
    ggplotly() %>%
    plotly::layout(legend = list(orientation = "h", x = 0.2, y = -0.4))

ggsave(survfit_png, filename = "img/014_churn_survical_fit.png")
htmlwidgets::saveWidget(survfit_html, "img/014_churn_survical_fit.html")


survfit_coxph <- survfit(model_coxph)
plot_customer_survival(survfit_coxph)

plot_customer_loss <- function(object_survfit) {
    
    g <- tidy(object_survfit) %>%
        mutate(customers_lost = 1 - estimate) %>%
        ggplot(aes(time, customers_lost, color = strata)) +
        geom_line(size = 1) +
        theme_minimal() +
        scale_color_manual(values = unname(palette.colors(palette = "Tableau"))) +
        scale_y_continuous(labels = scales::percent_format()) +
        labs(title = "Churn Problem", color = "Contract Type", 
             x = "Days After Purchase", y = "Percentage of Customers Lost")  +
        theme(legend.key=element_blank(),
              legend.position = "bottom") 
    g
    #ggplotly(g)
}

plot_customer_loss(survfit_simple)

plot_customer_loss(survfit_coxph)




# 6.0 PREDICTION with Survial Models ----

# 6.1 Cox PH - Produces Theoretical Hazard Ratio ----
# Not a great approach when you want to predict churn but it is possible. 
predict(model_coxph, newdata = train_tbl, type = "expected") %>%
    tibble(.pred = .) %>%
    bind_cols(train_tbl)

# 6.2 Survival Regression w/ Parsnip ----
model_survreg <- parsnip::survival_reg(mode = "censored regression", dist = "weibull") %>%
    set_engine("survival", control = survreg.control(maxiter=500)) %>%
    fit.model_spec(Surv(I(tenure + 1), Churn_Yes) ~ . - Contract + strata(Contract), data = train_tbl)

model_survreg$fit %>% tidy()

model_survreg$fit %>% survfit() # Get an error (not as convenient as CoxPH for getting survival curves)

predict(model_survreg, new_data = train_tbl) %>%
    bind_cols(train_tbl %>% select(Churn_Yes, everything()))


# SUMMARY:
# 6.1 CoxPH 
#   - Let's us use multivariate regression
# 6.2 Survival Curve
#   - Curves give us a clear indication of how cohorts churn
#   - We saw that if someone is Month-to-Month Contract, that group is 48% risk of leaving in first 50 days
# 6.3 Survival Regression 
#   - Gives us estimated time, but can be quite innaccurate

# CONCLUSION:
# 1. Survival curves help understand time-dependent churn rates
# 2. NEED Better Method that Describes Each Individual Accurately --> Machine Learning


# 7.0 MACHINE LEARNING FOR CHURN RISK ----

# 7.1 H2O ----
# - Use H2O to Develop ML Models 

check_h2o_version("3.24.0.5")

h2o.init()

model_h2o <- h2o.loadModel("h2o_model/StackedEnsemble_BestOfFamily_AutoML_20190715_084457")

model_h2o

predictions_tbl <- customer_churn_tbl %>%
    as.h2o() %>%
    h2o.predict(model_h2o, newdata = .) %>%
    as_tibble() %>%
    bind_cols(train_tbl %>% select(Churn_Yes, everything()))

predictions_tbl


# 7.2 LIME ----
# - Use LIME to Explain Locally (Why is Customer 7590-VHVEG Predicted  65% Probability to Leave?)

# LIME Explanation for first person
predictions_tbl %>%
    slice(1) %>% 
    glimpse()

lime_explanation <- read_rds("h2o_model/lime_explanation.rds")

plot_features_interactive <- function(explanation) {
    g <- explanation %>%
        as_tibble() %>%
        filter(label == "Yes") %>%
        plot_features()
    
    ggplotly(g)
}

plot_features_interactive(lime_explanation)


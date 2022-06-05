# A/B TESTING & STATISTICAL INFERENCE ----
# WEB TRAFFIC CONVERSION A/B TESTING ----

# LIBRARIES ----

# Core
library(tidyverse)
library(tidyquant)
library(lubridate)
library(plotly)

# EDA
library(skimr)

# Modeling & Inference
library(infer)

# Plotting & Conversion Calculations
source("utils_ab_testing_conversions.R")

# 1.0 Data ----

ab_data_raw_tbl <- read_csv("data/ab_data.csv")

ab_data_raw_tbl %>% skim()

# 2.0 EDA ----

# 2.1 Landing Page & Groups ----

ab_data_raw_tbl

ab_data_raw_tbl %>% calculate_conversion()

# Landing Page - Old vs New
ab_data_raw_tbl %>% calculate_conversion(landing_page)

# Group - Control vs Treatment
ab_data_raw_tbl %>% calculate_conversion(group)



# Landing Page & Group
ab_data_raw_tbl %>% calculate_conversion(group, landing_page)

ab_data_raw_tbl %>% 
    calculate_conversion(group, landing_page) %>%
    plot_conversion_heatmap(group, landing_page, 
                            labs_title = "A/B Testing Conversion Heatmap")
    


# 2.2 Conversion Over Time ----

comparison_1 <- c("control-old_page", "treatment-new_page")
# Comparison 2 is leakage. All treatment needed to go to the new page and all control needed to go to the old page. I will not use these records.
comparison_2 <- c("control-new_page", "treatment-old_page")

# Comparison 1: Control + Old Page vs Treatment + New Page
# - Majority of Traffic
#Visualize your data over time. You always want to look at spikes and any anomalies.  
ab_data_raw_tbl %>%
    calculate_conversion_vs_time(group, landing_page, unit = "1 day") %>%
    mutate(label = str_c(group, landing_page, sep = "-")) %>%
    filter(label %in% comparison_1) %>%
    plot_conversion_vs_time(color = label, loess = TRUE, loess_se = FALSE,
                            labs_title = "Conversion Over Time - Majority of Traffic")
# Both seem pretty flat. 

# Comparison 2: Control + New Page vs Treatment + Old Page
# - Minority of Traffic
ab_data_raw_tbl %>%
    calculate_conversion_vs_time(group, landing_page, unit = "1 day") %>%
    mutate(label = str_c(group, landing_page, sep = "-")) %>%
    filter(label %in% comparison_2) %>%
    plot_conversion_vs_time(color = label, loess = TRUE, loess_se = FALSE, loess_span = 0.6)



# 3.0 Feature Engineering ----

# 3.1 Combine Group & Landing Page
ab_data_tbl <- ab_data_raw_tbl %>%
    mutate(label = str_c(group, landing_page, sep = "-")) %>%
    filter(label %in% comparison_1) %>%
    select(user_id, timestamp, label, converted)


# 4.0 DISTRIBUTION ANALYSIS ----

ab_data_tbl %>%
    calculate_conversion_vs_time(label, unit = "day") %>%
    ggplot(aes(prop)) +
    geom_histogram(bins = 100) +
    facet_wrap(~ label)



# 5.0 DIFFERENCE IN MEANS ----

# - Assumption: Conversion rate is independent of date 
#   - This assumption is not valid for total web traffic, conversion counts, etc since date may influence
#   - This assumption is valid for conversion rates, which should be constant if visits are large enough

# - Hypothesis Testing: https://moderndive.com/9-hypothesis-testing.html
#   - Want to understand if mean of Conversions for Treatment-New Page is different than Control-Old Page
#   - Test for difference in means

# 5.1 Majority Traffic: Treatment_NewPage vs Control_OldPage ----

comparison_1_tbl <- ab_data_tbl %>%
    filter(label %in% comparison_1) %>%
    calculate_conversion_vs_time(label, unit = "1 day")

comparison_1_tbl %>% plot_conversion_vs_time(color = label, loess_se = FALSE)

comparison_1_tbl %>%
    ggplot(aes(prop)) +
    geom_histogram(bins = 20) +
    facet_wrap(~ label, ncol = 1)

# No Permutation - Get the mean difference in daily conversion rate
comp_1_diff_means <- comparison_1_tbl %>%
    
    # Infer
    specify(prop ~ label) %>%
    calculate(stat  = "diff in means", 
              order = c("treatment-new_page", "control-old_page"))

comp_1_diff_means
# New page decreases conversion rate!

# Permutation - Find *P-Value* in Mean Difference ----
# - Why? Cannot trust a point estimate - Sample Variation
# - Goal: Prove difference in mean conversion rate between Treatment-New Page vs Control-Old Page
# - Use Permutation = Sample *without* replacement
# - Calculate P-Value of Hypothesis:
# - H0 (Null): Improvement is <= 1%
# - H1 (Alt): Improvement is > 1%

comp_1_diff_means_permuted_tbl <- comparison_1_tbl %>%
    
    # Infer
    specify(prop ~ label) %>%
    hypothesize(null = "independence") %>%
    generate(reps = 1000, type = "permute") %>%
    calculate(stat = "diff in means", 
              order = c("treatment-new_page", "control-old_page")) 

# what is the probability that new function increases sign ups by 1%
comp_1_diff_means_permuted_tbl %>%
    visualize() +
    shade_p_value(obs_stat = 0.01, direction = "right")

comp_1_diff_means_permuted_tbl %>% 
    get_p_value(obs_stat = 0.0, direction = "right")
    
# Bootstrap - Find True Distribution & Confidence Intervals ----
# - Permutation - Gave us P-Value, reject difference in mean
# - What about distribution of mean differences?
# - Goal: Confidence Intervals & Distribution - Use Bootstrap
# - Sample *with* replacement
comp_1_diff_bootstrap_tbl <- comparison_1_tbl %>%
    
    # Infer
    specify(prop ~ label) %>%
    
    # Remove Hypothesis:
    # hypothesize(null = "independence") %>%
    
    # Change generate(): type = "boostrap"
    generate(reps = 1000, type = "bootstrap") %>%
    calculate(stat = "diff in means", 
              order = c("treatment-new_page", "control-old_page")) 

bootstrap_ci_tbl <- comp_1_diff_bootstrap_tbl %>%
    get_confidence_interval(level = 0.95, type = "percentile")

comp_1_diff_bootstrap_tbl %>%
    visualize() +
    theme_minimal() + 
    shade_confidence_interval(endpoints = bootstrap_ci_tbl) +
    geom_vline(xintercept = comp_1_diff_means$stat, color = "red")
# 95% confidence interval is around 0 and most of probability distribution is negative. 
# No go. 




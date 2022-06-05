# A/B TESTING & STATISTICAL INFERENCE ----
# WEB TRAFFIC CONVERSION ----

calculate_conversion <- function(data, ...) {
    
    groups_expr <- enquos(...)
    
    data %>%
        group_by(!!! groups_expr) %>%
        summarize(
            count      = n(),
            conversion = sum(converted),
            prop       = conversion / count
        ) %>%
        ungroup()
    
}

calculate_conversion_vs_time <- function(data, ..., unit = "day") {
    
    groups_expr <- enquos(...)
    
    data %>%
        mutate(date = floor_date(timestamp, unit = unit)) %>%
        group_by(!!!groups_expr, date) %>%
        summarize(
            count      = n(),
            conversion = sum(converted),
            prop       = conversion / count
        ) %>%
        ungroup()
    
}

plot_conversion_vs_time <- function(data, ..., 
                                    loess = TRUE, loess_span = 0.3, loess_se = TRUE,
                                    labs_title = "Conversion Over Time", 
                                    labs_y = "Conversion (%)", labs_x = "") {
    
    g <- data %>%
        ggplot(aes(date, prop, ...)) +
        geom_line() +
        expand_limits(y = 0) +
        scale_y_continuous(labels = scales::percent_format()) +
        theme_minimal() +
        scale_color_brewer(type = "qual", palette = 3) +
        labs(title = labs_title, y = labs_y, x = labs_x) 
    
    if (loess) g <- g + geom_smooth(method = "loess", span = loess_span, se = loess_se)
    
    ggplotly(g)
}

plot_conversion_heatmap <- function(data, .x, .y, 
                                    labs_title = "Conversion Heatmap") {

    x_expr <- enquo(.x)
    y_expr <- enquo(.y)
    
    
    g <- data %>%
        mutate(label = str_glue("Conv Pct: {scales::percent(prop, accuracy = 0.1)}
                                Converted: {scales::comma(conversion)}
                                Total: {scales::comma(count)}")) %>%
        
        ggplot(aes(x = !!x_expr, y = !!y_expr, fill = prop)) +
        geom_tile() +
        geom_text(aes(label = label), color = "white") +
        theme_minimal() +
        theme(legend.position = "none") +
        labs(title = labs_title)
    
    # ggplotly(g, tooltip = "text")
    
    g
}


    
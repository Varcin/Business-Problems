
check_h2o_version <- function (h2o_version_model = "3.24.0.5") {
    
    # Check H2O Version
    h2o_version_current <- installed.packages() %>% 
        as_tibble() %>%
        filter(Package == "h2o") %>%
        select(Package, Version) %>%
        pull(Version)
    
    if (h2o_version_current != h2o_version_model) {
        
        msg <- readline(prompt="Current version of H2O is incompatible. Would you like to install (y/n): ")
        
        if (msg == "n") {
            # The following two commands remove any previously installed H2O packages for R.
            if ("package:h2o" %in% search()) 
                detach("package:h2o", unload=TRUE) 
            if ("h2o" %in% rownames(installed.packages())) 
                remove.packages("h2o") 
            # Repo for H2O Version 3.24.0.5
            install.packages("h2o", type="source", repos="https://h2o-release.s3.amazonaws.com/h2o/rel-yates/5/R")
        } else {
          warning("H2O Model will fail on load. Please skip the h2o section.")   
        }
        
    } else {
        message("Current version of H2O is OK.")
    }
    
}


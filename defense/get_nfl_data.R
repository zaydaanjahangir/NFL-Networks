# Script to fetch NFL play-by-play data using nflfastR
if (!require("nflfastR")) install.packages("nflfastR")
library(nflfastR)
cat("Fetching NFL play-by-play data for 2022 season...\n")
pbp_2022 <- nflfastR::load_pbp(2022)

output_file <- "pbp_2022.csv"
cat(paste("Saving data to", output_file, "...\n"))
write.csv(pbp_2022, output_file, row.names = FALSE)

cat(paste("Successfully saved", nrow(pbp_2022), "plays to", output_file, "\n"))
cat("Sample columns: game_id, play_id, epa, play_description\n") 
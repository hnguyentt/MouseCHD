
setwd("~/DATA/INCEPTION_2020-CHD/Mice/projects/MouseCHD/analysis")
install.packages("rjson")
install.packages("./resrc/nVennR_0.2.3.tar.gz", repos = NULL, type = "source") # https://cran.r-project.org/src/contrib/Archive/nVennR/
library("rjson")
library(nVennR)

DATADIR <- "/home/hnguyent/DATA/INCEPTION_2020-CHD/Mice/PAPER"
outdir <- file.path(DATADIR, "/FIGURES/revision1")
if (!dir.exists(outdir)) {
     dir.create(outdir, recursive = TRUE)
}

# Initial cohort
kingdom <- fromJSON(paste(readLines(file.path(DATADIR, "METADATA/VennData_initial_cohort.json"), warn = FALSE), collapse = ""))

plotVenn(kingdom,
     borderWidth = 2, fontScale = 2,
     outFile = file.path(outdir, "Venn_initial.svg")
)

# Prospective cohort
kingdom <- fromJSON(paste(readLines(file.path(DATADIR, "METADATA/VennData_prospective_cohort.json"), warn = FALSE), collapse = ""))

plotVenn(kingdom,
     borderWidth = 2, fontScale = 2,
     outFile = file.path(outdir, "Venn_prospective.svg")
)

# Divergent cohort
kingdom <- fromJSON(paste(readLines(file.path(DATADIR, "METADATA/VennData_divergent_cohort.json"), warn = FALSE), collapse = ""))

plotVenn(kingdom,
     borderWidth = 2, fontScale = 2,
     outFile = file.path(outdir, "Venn_divergent.svg")
)
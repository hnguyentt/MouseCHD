#########
# SETUP #
#########
# Adjust paths here for your own setup. Later, you don't need to modify any following codes.
setwd("~/DATA/INCEPTION_2020-CHD/Mice/projects/MouseCHD/analysis")
DATADIR <- "/home/hnguyent/DATA/INCEPTION_2020-CHD/Mice/PAPER"
outdir <- file.path(DATADIR, "/FIGURES/revision1")
if (!dir.exists(outdir)) {
     dir.create(outdir, recursive = TRUE)
}

##############
# UpSet plot #
##############
install.packages("ComplexUpset")

library(ggplot2)
library(ComplexUpset)

# Initial cohort
df <- read.csv(file.path(DATADIR, "METADATA/UpSet_initial_cohort.csv"))

svg(filename = file.path(outdir, "upset_initial.svg"))
upset(
    df,
    c("Apex.malposition", "Atrial.situs.defects", "Malposition.of.the.great.arteries", "Septal.defects", "Ventricle.malposition"),
    name="combinations",
    width_ratio=0.2,
    group_by='degree',
    matrix=(
     intersection_matrix(geom=geom_point(shape="circle filled", size=3))
     + scale_color_manual(
          values=c("Apex.malposition"="#e6194bff", "Atrial.situs.defects"="#3cb44bff", "Malposition.of.the.great.arteries"="#ffe119ff", "Septal.defects"="#0082c8ff", "Ventricle.malposition"="#f58231ff"),
          guide=guide_legend(override.aes=list(shape="circle"))
     )
    ),
    queries=list(
     upset_query(set="Apex.malposition", fill="#e6194bff"),
     upset_query(set="Atrial.situs.defects", fill="#3cb44bff"),
     upset_query(set="Malposition.of.the.great.arteries", fill="#ffe119ff"),
     upset_query(set="Septal.defects", fill="#0082c8ff"),
     upset_query(set="Ventricle.malposition", fill="#f58231ff")
    ),
    labeller=ggplot2::as_labeller(c(
        'Apex.malposition'='AM',
        'Atrial.situs.defects'='ASD',
        'Malposition.of.the.great.arteries'='MGA',
        'Septal.defects'='SD',
        'Ventricle.malposition'='VM'
    ))
)
dev.off()

# Prospective cohort
df <- read.csv(file.path(DATADIR, "METADATA/UpSet_prospective_cohort.csv"))

svg(filename = file.path(outdir, "upset_prospective.svg"))
upset(
    df,
    c("Apex.malposition", "Atrial.situs.defects", "Malposition.of.the.great.arteries", "Septal.defects", "Ventricle.malposition"),
    name="combinations",
    width_ratio=0.2,
    group_by='degree',
    matrix=(
     intersection_matrix(geom=geom_point(shape="circle filled", size=3))
     + scale_color_manual(
          values=c("Apex.malposition"="#e6194bff", "Atrial.situs.defects"="#3cb44bff", "Malposition.of.the.great.arteries"="#ffe119ff", "Septal.defects"="#0082c8ff", "Ventricle.malposition"="#f58231ff"),
          guide=guide_legend(override.aes=list(shape="circle"))
     )
    ),
    queries=list(
     upset_query(set="Apex.malposition", fill="#e6194bff"),
     upset_query(set="Atrial.situs.defects", fill="#3cb44bff"),
     upset_query(set="Malposition.of.the.great.arteries", fill="#ffe119ff"),
     upset_query(set="Septal.defects", fill="#0082c8ff"),
     upset_query(set="Ventricle.malposition", fill="#f58231ff")
    ),
    labeller=ggplot2::as_labeller(c(
        'Apex.malposition'='AM',
        'Atrial.situs.defects'='ASD',
        'Malposition.of.the.great.arteries'='MGA',
        'Septal.defects'='SD',
        'Ventricle.malposition'='VM'
    ))
)
dev.off()

# divergent cohort
df <- read.csv(file.path(DATADIR, "METADATA/UpSet_divergent_cohort.csv"))

svg(filename = file.path(outdir, "upset_divergent.svg"))
upset(
    df,
    c("Apex.malposition", "Atrial.situs.defects", "Malposition.of.the.great.arteries", "Septal.defects", "Ventricle.malposition", "Situs.inversus.totalis"),
    name="combinations",
    width_ratio=0.2,
    group_by='degree',
    matrix=(
     intersection_matrix(geom=geom_point(shape="circle filled", size=3))
     + scale_color_manual(
          values=c(
               "Apex.malposition"="#e6194bff", 
               "Atrial.situs.defects"="#3cb44bff", 
               "Malposition.of.the.great.arteries"="#ffe119ff", 
               "Septal.defects"="#0082c8ff", 
               "Ventricle.malposition"="#f58231ff", 
               "Situs.inversus.totalis"="#911eb4ff"),
          guide=guide_legend(override.aes=list(shape="circle"))
     )
    ),
    queries=list(
     upset_query(set="Apex.malposition", fill="#e6194bff"),
     upset_query(set="Atrial.situs.defects", fill="#3cb44bff"),
     upset_query(set="Malposition.of.the.great.arteries", fill="#ffe119ff"),
     upset_query(set="Septal.defects", fill="#0082c8ff"),
     upset_query(set="Ventricle.malposition", fill="#f58231ff"),
     upset_query(set="Situs.inversus.totalis", fill="#911eb4ff")
    ),
    labeller=ggplot2::as_labeller(c(
        'Apex.malposition'='AM',
        'Atrial.situs.defects'='ASD',
        'Malposition.of.the.great.arteries'='MGA',
        'Septal.defects'='SD',
        'Ventricle.malposition'='VM',
        'Situs.inversus.totalis'='SIT'
    ))
)
dev.off()

################
# Venn Diagram #
################
install.packages("rjson")
install.packages("./resrc/nVennR_0.2.3.tar.gz", repos = NULL, type = "source") # https://cran.r-project.org/src/contrib/Archive/nVennR/
library("rjson")
library(nVennR)

# Initial cohort
kingdom <- fromJSON(paste(readLines(file.path(DATADIR, "METADATA/VennData_initial_cohort.json"), warn = FALSE), collapse = ""))

venn <- plotVenn(kingdom,
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

###############
# Circos plot #
###############
library(circlize)

# Initial cohort
df <- read.csv(file.path(DATADIR, "METADATA/circos_initial_cohort.csv"))

svg(filename = file.path(outdir, "circos_initial.svg"))
diseases_order <- c(
     "Apex malposition",
     "Atrial situs defects",
     "Malposition of the great arteries",
     "Septal defects",
     "Ventricle malposition"
)

grid.col = disease_order <- c(
     "Apex malposition" = "#e6194bff",
     "Atrial situs defects" = "#3cb44bff",
     "Malposition of the great arteries" = "#ffe119ff",
     "Septal defects" = "#0082c8ff",
     "Ventricle malposition" = "#f58231ff"
)

chordDiagram(df, order = diseases_order, directional = 1, grid.col = grid.col,
             annotationTrack = "grid",
             preAllocateTracks = list(track.height = max(strwidth(unlist(dimnames(df))))))
circos.track(track.index = 1, panel.fun = function(x, y) {
  xlim = get.cell.meta.data("xlim")
  xplot = get.cell.meta.data("xplot")
  ylim = get.cell.meta.data("ylim")
  sector.name = get.cell.meta.data("sector.index")
  num = sum(df[df$from == sector.name,]$value1) + sum(df[df$to == sector.name,]$value2)
  
  circos.text(mean(xlim), ylim[1], paste(sector.name, num, sep="\n") , facing = "inside", 
              niceFacing = TRUE, adj = c(0.5, 0), col= "black")
}, bg.border = NA)

dev.off()

# Prospective cohort
df <- read.csv(file.path(DATADIR, "METADATA/circos_prospective_cohort.csv"))

svg(filename = file.path(outdir, "circos_prospective.svg"))
diseases_order <- c(
     "Apex malposition",
     "Atrial situs defects",
     "Malposition of the great arteries",
     "Septal defects",
     "Ventricle malposition"
)

grid.col = disease_order <- c(
     "Apex malposition" = "#e6194bff",
     "Atrial situs defects" = "#3cb44bff",
     "Malposition of the great arteries" = "#ffe119ff",
     "Septal defects" = "#0082c8ff",
     "Ventricle malposition" = "#f58231ff"
)

chordDiagram(df, order = diseases_order, directional = 1, grid.col = grid.col,
             annotationTrack = "grid",
             preAllocateTracks = list(track.height = max(strwidth(unlist(dimnames(df))))))
circos.track(track.index = 1, panel.fun = function(x, y) {
  xlim = get.cell.meta.data("xlim")
  xplot = get.cell.meta.data("xplot")
  ylim = get.cell.meta.data("ylim")
  sector.name = get.cell.meta.data("sector.index")
  num = sum(df[df$from == sector.name,]$value1) + sum(df[df$to == sector.name,]$value2)
  
  circos.text(mean(xlim), ylim[1], paste(sector.name, num, sep="\n") , facing = "inside", 
              niceFacing = TRUE, adj = c(0.5, 0), col= "black")
}, bg.border = NA)

dev.off()

# Divergent cohort
df <- read.csv(file.path(DATADIR, "METADATA/circos_divergent_cohort.csv"))

svg(filename = file.path(outdir, "circos_divergent.svg"))
diseases_order <- c(
     "Apex malposition",
     "Atrial situs defects",
     "Malposition of the great arteries",
     "Septal defects",
     "Situs inversus totalis",
     "Ventricle malposition"
)

grid.col = disease_order <- c(
     "Apex malposition" = "#e6194bff",
     "Atrial situs defects" = "#3cb44bff",
     "Malposition of the great arteries" = "#ffe119ff",
     "Septal defects" = "#0082c8ff",
     "Ventricle malposition" = "#f58231ff",
     "Situs inversus totalis" = "#911eb4ff"
)

chordDiagram(df, order = diseases_order, directional = 1, grid.col = grid.col,
             annotationTrack = "grid",
             preAllocateTracks = list(track.height = max(strwidth(unlist(dimnames(df))))))
circos.track(track.index = 1, panel.fun = function(x, y) {
  xlim = get.cell.meta.data("xlim")
  xplot = get.cell.meta.data("xplot")
  ylim = get.cell.meta.data("ylim")
  sector.name = get.cell.meta.data("sector.index")
  num = sum(df[df$from == sector.name,]$value1) + sum(df[df$to == sector.name,]$value2)
  
  circos.text(mean(xlim), ylim[1], paste(sector.name, num, sep="\n") , facing = "inside", 
              niceFacing = TRUE, adj = c(0.5, 0), col= "black")
}, bg.border = NA)

dev.off()

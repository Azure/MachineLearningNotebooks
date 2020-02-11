#' Copyright(c) Microsoft Corporation.
#' Licensed under the MIT license.

library(azuremlsdk)
library(optparse)
library(caret)

options <- list(
  make_option(c("-d", "--data_folder"))
)

opt_parser <- OptionParser(option_list = options)
opt <- parse_args(opt_parser)

paste(opt$data_folder)

accidents <- readRDS(file.path(opt$data_folder, "accidents.Rd"))
summary(accidents)

mod <- glm(dead ~ dvcat + seatbelt + frontal + sex + ageOFocc + yearVeh + airbag + occRole, family = binomial, data = accidents)
summary(mod)
predictions <- factor(ifelse(predict(mod) > 0.1, "dead", "alive"))
conf_matrix <- confusionMatrix(predictions, accidents$dead)
message(conf_matrix)

log_metric_to_run("Accuracy", conf_matrix$overall["Accuracy"])

output_dir = "outputs"
if (!dir.exists(output_dir)) {
    dir.create(output_dir)
}
saveRDS(mod, file = "./outputs/model.rds")
message("Model saved")

#' Copyright(c) Microsoft Corporation.
#' Licensed under the MIT license.

library(optparse)

options <- list(
  make_option(c("-d", "--data_folder")),
  make_option(c("--output_folder"))
  
)

opt_parser <- OptionParser(option_list = options)
opt <- parse_args(opt_parser)

paste(opt$data_folder)

accidents <- readRDS(file.path(opt$data_folder, "accidents.Rd"))
summary(accidents)

mod <- glm(dead ~ dvcat + seatbelt + frontal + sex + ageOFocc + yearVeh + airbag  + occRole, family=binomial, data=accidents)
summary(mod)
predictions <- factor(ifelse(predict(mod)>0.1, "dead","alive"))
accuracy <- mean(predictions == accidents$dead)

# make directory for output dir
output_dir = opt$output_folder
if (!dir.exists(output_dir)){
  dir.create(output_dir)
}

# save model
model_path = file.path(output_dir, "model.rds")
saveRDS(mod, file = model_path)
message("Model saved")
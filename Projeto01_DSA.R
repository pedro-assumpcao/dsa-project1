library(pacman)

p_load(data.table)
p_load(ggplot2)
p_load(readxl)
p_load(DataExplorer)
p_load(janitor)
p_load(naniar)
p_load(vip)
p_load(tidymodels)
p_load(e1071)
p_load(esquisse)
p_load(robustbase)
p_load(outliers)
p_load(plotly)
p_load(corrplot)
p_load(Boruta)



#Reading Data ----

setwd('C:\\Users\\pedro_jw08iyg\\OneDrive\\Área de Trabalho\\DSA\\Projetos\\Big Data Analytics com R e Microsoft Azure Machine Learning\\Projeto1')

raw_dataframe = readxl::read_excel('data/FEV-data-Excel.xlsx')
setDT(raw_dataframe)


#Cleaning headers
raw_dataframe = janitor::clean_names(raw_dataframe)



#1) Data Overview ----

tibble::glimpse(raw_dataframe)

summary(raw_dataframe)


#removing non-informative variables
cleaned_dataframe = copy(raw_dataframe)
cleaned_dataframe[,car_full_name:=NULL]
cleaned_dataframe[,model:=NULL]

#converting to factor
cleaned_dataframe[,number_of_doors:=as.factor(number_of_doors)]
cleaned_dataframe[,number_of_seats:=as.factor(number_of_seats)]
cleaned_dataframe[,tire_size_in:=as.factor(tire_size_in)]



# 2) Dealing with Missing Values ----

#overview
naniar::vis_miss(raw_dataframe)

#by column
naniar::miss_var_summary(raw_dataframe)

#plot percentage by column
naniar::gg_miss_var(raw_dataframe)


#by row
missing_rows = naniar::miss_case_summary(raw_dataframe)
setDT(missing_rows)
missing_rows = missing_rows[n_miss!=0]


#2% of total data is mising
#6 features have missing values, raging between 2% and 17% by columns



raw_dataframe_missing = copy(raw_dataframe)
raw_dataframe_missing = naniar::bind_shadow(raw_dataframe_missing)
setDT(raw_dataframe_missing)




#4) EDA and Dealing with Outliers ----

#cleaning outliers
cleaned_dataframe = cleaned_dataframe[width_cm < 220]
cleaned_dataframe = cleaned_dataframe[length_cm > 300]
cleaned_dataframe = cleaned_dataframe[engine_power_km < 700]
cleaned_dataframe = cleaned_dataframe[maximum_load_capacity_kg < 800]
cleaned_dataframe = cleaned_dataframe[acceleration_0_100_kph_s < 12.5]


#Bivariate Analysis - Correlation Matrix
corrplot(cor(cleaned_dataframe[,.SD,.SDcols = is.numeric],method = 'spearman', use = "complete.obs"), method = 'square', order = 'FPC', type = 'lower', diag = FALSE)


#5) Feature Selection ----


cleaned_dataframe_without_missing = cleaned_dataframe[complete.cases(cleaned_dataframe)]

#performing Boruta
temp_boruta = Boruta::Boruta(mean_energy_consumption_k_wh_100_km~., data = cleaned_dataframe_without_missing, doTrace = 2)
rm(cleaned_dataframe_without_missing)

#visualizing results
plot(temp_boruta)

#getting selected variables
selected_variables = Boruta::getSelectedAttributes(temp_boruta, withTentative = F)

#adding target variable to the filtering vector
selected_variables = c('mean_energy_consumption_k_wh_100_km',selected_variables)

#filtering columns
cleaned_dataframe = cleaned_dataframe[, ..selected_variables, with=FALSE]
  

#removing missing values
cleaned_dataframe = cleaned_dataframe[!is.na(mean_energy_consumption_k_wh_100_km)]
cleaned_dataframe = cleaned_dataframe[!is.na(acceleration_0_100_kph_s)]

#removing high-correlated variable
cleaned_dataframe[,minimal_empty_weight_kg:=NULL]


# 6) Data Resampling ----

set.seed(123)
cleaned_dataframe_split = rsample::initial_split(data = cleaned_dataframe,
                                                 prop = 0.70,
                                                 strata = drive_type) 

cleaned_dataframe_training = cleaned_dataframe_split |> training()
cleaned_dataframe_testing = cleaned_dataframe_split |> testing()


#7) Model Specification ----
baseline_model = linear_reg() |> 
  set_engine("lm") |>
  set_mode('regression')



#8) Feature Engineering ----

recipe_baseline = recipes::recipe(mean_energy_consumption_k_wh_100_km~.,
                                  data = cleaned_dataframe_training) |>
  # step_cut(acceleration_0_100_kph_s,breaks = 3) |>
  step_corr(all_numeric(),-all_outcomes(), threshold = 0.8)  |>
  #step_YeoJohnson(all_outcomes()) |>
  step_normalize(all_numeric(),-all_outcomes() ) |>
  step_dummy(all_nominal(),-all_outcomes())


#9) Recipe Training ----
  

recipe_prep_baseline = recipe_baseline |> 
  prep(training = cleaned_dataframe_training)

#10) Preprocess training data ----

cleaned_dataframe_training_prep = recipe_prep_baseline |>
  recipes::bake(new_data = NULL)


#11) Preprocess test data ----

cleaned_dataframe_testing_prep = recipe_prep_baseline |> 
  recipes::bake(new_data = cleaned_dataframe_testing)


#12) Model fitting ----

baseline_model_fit = baseline_model |>
  parsnip::fit(mean_energy_consumption_k_wh_100_km ~ .,
      data = cleaned_dataframe_training_prep)


parsnip::tidy(baseline_model_fit)
parsnip::glance(baseline_model_fit)

#13) Predictions on Test Data and metrics

predictions = predict(baseline_model_fit,
                           new_data = cleaned_dataframe_testing_prep)

cleaned_dataframe_testing_prep = bind_cols(cleaned_dataframe_testing_prep,predictions)

#RMSE
cleaned_dataframe_testing_prep |>
  yardstick::rmse(truth = mean_energy_consumption_k_wh_100_km , .pred)

#R-squared
cleaned_dataframe_testing_prep |>
  yardstick::rsq(truth = mean_energy_consumption_k_wh_100_km , .pred)


install.packages('corrgram')
install.packages('corrplot')
install.packages('caTools')
library(dplyr)
library(ggplot2)
library(ggthemes)
library(corrgram)
library(corrplot)
library(caTools)
# student Math performance data set
df <- read.csv('C:\\Users\\aabha.DESKTOP-HG6KK17\\Downloads\\R-Course-HTML-Notes\\R-Course-HTML-Notes\\R-for-Data-Science-and-Machine-Learning\\Machine Learning with R\\student-mat.csv', sep=';')

head(df)
 
summary(df)

# G1 is the first period grade, G2 is the second period grade, G3 is the final period grade
# we are going to predict the final grade

# DATA CLEANING

# check for missing values; # False means we have no missing values

any(is.na(df))


# checking if categorical variables are of factor type
str(df)

# variables Medu, Fedu should be of factor, but we will revisit this during model building process

# EXPLORATORY DATA ANALYSIS

# calling only the numeric variables in df
num.cols <- sapply(df, is.numeric)

# checking for correlation between numeric variables
cor.data <- cor(df[, num.cols])
print(cor.data)

# we will visualize this correlation data so that its easier to understand

print(corrplot(cor.data, method = 'color'))

# There is a high correlation between G1, G2 and G3. That makes sense because if a student had a good grade in first
# period and second period, will probably get a good grade in the final period also.
# Inverse correlation between failures variable and G1, G2, and G3 also makes sense.

# with corrgram we can pass in the dataframe directly
# we are specifying the lower panel to be regular shaded box, upper panel to be pie charts, full pie-chart would
# indicate perfect correlation of 1; how filled the correlation pie chart is how correlated the variables are;
# blue means positive correlation, red means negative correlation
corrgram(df, order = TRUE, lower.panle = panel.shade, upper.panel = panel.pie, text.panel = panel.txt)

# plot G3 variable 
print(ggplot(df, aes(x=G3)) + geom_histogram(bins = 20, alpha = 0.5, fill = 'blue'))

# SPLITTING THE DATA IN TRAIN AND TEST SET
# caTools package makes splitting the data in train and test set very easy

# set a seed
set.seed(101)

#split up sample; use the column that you are trying to predict; split ratio is 70% for training data and 30% for 
# testing
sample <- sample.split(df$G3,SplitRatio = 0.7)

# 70% of data goes to train dataset; subset df where sample == TRUE
train <- subset(df, sample == TRUE)
# 30% of data goes to test dataset
test <- subset(df, sample == FALSE)

# BUILDING THE MODEL
model <- lm(G3 ~ ., data = train)

# PREDICTIONS
G3.predictions <- predict(model, test)

results <- cbind(G3.predictions, test$G3)
colnames(results) <- c('predicted', 'actual')
results <- as.data.frame(results)
print(head(results))

# Taking care of negative values in predictions 
to_zero <- function(x){
  if (x <0) {
    return(0)
  }else{
    return(x)
  }
}

# Apply zero function
results$predicted <- sapply(results$predicted, to_zero)

# Mean SQUARED ERROR
mse <- mean( (results$actual - results$predicted)^2 )
print('MSE')
print(mse)

# RMSE
print("Squared root of MSE")
print(mse^0.5)

# MSE and RMSE give you an idea of how off you are

# 
SSE <- sum( (results$predicted - results$actual)^2 )
SST <- sum( (mean(df$G3) - results$actual)^2 )

R2 <- 1-SSE/SST
print('R2')
print(R2)
# we are explaining 80% variance

# Interpret the model
print(summary(model))

# Residuals are difference between the actual values of G3 and the predicted values of G3 from the model; we want the 
# distribution of residuals to be normal when plotted because if residuals are normally distributed this indicates 
# that the mean of the difference between our predictions and the actual values is close to zero; we want to 
# minimize the residuals value

res <- residuals(model)

res <- as.data.frame(res)

ggplot(res, aes(res)) + geom_histogram(fill='blue', alpha = 0.5)

# Coefficients has all the independent variables; estimate is the value of slope calculated by the regression; It is more 
# useful if we normalize our data; When we have normalized data we can compare the estimate to each other otherwise no;
# t value is used to calculate the p-value
# Pr(>|t|) - probability that the independent variable is not relevant - we want this to be as small as possible - 
# more stars next to Pr(>|t|) means higher significance. One star or period means low significance
# the stars indicate the probablity that the variable is not relevant, so we want the Pr(>|t|) to be as low as possible

# R-squared : is a metric for evaluating the Goodness of Fit of the model. Higher is better. High R-square means 
# higher correlation; however correlation does not always imply causation

plot(model)



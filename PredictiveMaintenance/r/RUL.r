
install.packages('gbm')
library(gbm)

train_raw <- read.table("../data/train_FD001.txt", 
    sep=" ", 
    colClasses=c(rep("numeric", 2), rep("double", 24), rep("NULL", 2)),
    col.name=c("id", "cycle", "setting1", "setting2", "setting3",
               "s1", "s2", "s3", "s4", "s5", "s6",
               "s7", "s8", "s9", "s10", "s11", "s12",
               "s13", "s14", "s15", "s16", "s17", "s18",
               "s19", "s20", "s21", "na", "na")
)
head(train_raw)

train_maxcycle <- setNames(aggregate(cycle~id,train_raw,max), c("id", "max"))
train_labeled <- merge(train_raw,train_maxcycle,by=c("id"))
train_labeled$RUL <- train_labeled$max - train_labeled$cycle
train_df <- train_labeled[, c("id", "cycle", "s9", "s11", "s14", "s15", "RUL")]
head(train_df)

test_raw <- read.table("../data/test_FD001.txt", 
    sep=" ", 
    colClasses=c(rep("numeric", 2), rep("double", 24), rep("NULL", 2)),
    col.name=c("id", "cycle", "setting1", "setting2", "setting3",
               "s1", "s2", "s3", "s4", "s5", "s6",
               "s7", "s8", "s9", "s10", "s11", "s12",
               "s13", "s14", "s15", "s16", "s17", "s18",
               "s19", "s20", "s21", "na", "na")
)
head(test_raw)

test_maxcycle <- aggregate(cycle~id,test_raw,max)
test_maxcycle_only <- merge(test_maxcycle,test_raw)[, c("id", "cycle", "s9", "s11", "s14", "s15")]
test_ordered = test_maxcycle_only[order(test_maxcycle_only$id), ]
head(test_ordered)

rul_df <- read.table("../data/RUL_FD001.txt", 
    colClasses=c("numeric"),
    col.name=c("RUL")
)
head(rul_df)

test_df <- cbind(test_ordered, rul_df)
head(test_df)

formula <- as.formula("RUL ~ cycle + s9 + s11 + s14 + s15")
gbt <- gbm(
    formula = formula, 
    data = train_df, 
    shrinkage = 0.2, 
    n.trees = 100, 
    distribution = "gaussian" 
    )

predictions <- predict(object = gbt, newdata = test_df, n.trees = 100)

evaluate_model <- function(observed, predicted) {
  se <- (observed - predicted)^2
  rmse <- sqrt(mean(se))
  metrics <- c("Root Mean Squared Error" = rmse)
  return(metrics)
}

rmse <- evaluate_model(observed = test_df$RUL, predicted = predictions)
rmse

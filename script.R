############### Preprocessing ################

# Open data.csv
raw_data <- read.csv("Data.csv")

# Discard "adj.close" --> Only selects those relevant columns
data <- raw_data[, c("Open", "High", "Low", "Close", "Volume")]

# Dicard row where Volume is NA
data <- data[complete.cases(data[,c("Volume")]),]
data_len <- nrow(data)

next_non_na <- function(data, col, min_idx) {
  stopifnot(is.data.frame(data))
  stopifnot(is.numeric(min_idx))
  
  i = min_idx
  data_len <- nrow(data)
  while (i < data_len + 1) {
    if (!is.na(data[i, col])) {
      return(i)
    }
    i <- i + 1
  }
  return(-1) # not found
}

# Handle missing data
for (col in names(data)) {
  # find first non-NA row
  first_idx = next_non_na(data, col, 1)
  if (first_idx > 1) {
    # fill index 1 .. (first_idx - 1) with data[first_idx]
    for (i in 1:(first_idx)) {
      data[i, col] <- data[first_idx, col]
    }
  }
  left = first_idx + 1
  right = first_idx + 1
  max_idx = first_idx
  
  # middle
  while (left < data_len + 1 && right > 0) {
    
    left = next_non_na(data, col, right)
    right = next_non_na(data, col, left + 1)
    if (right > left + 1) {
      # fill index (left + 1) .. (right - 1) with mean
      mean_val <- mean(c(data[left, col], data[right, col]))
      for (i in (left + 1):(right - 1)) {
        data[i, col] <- mean_val
      }
    }
    max_idx <- max(max_idx, right)
  }
  
  # fill in last data
  if (max_idx < data_len) {
    for (i in (max_idx + 1):(data_len)) {
      data[i, col] <- data[max_idx, col]
    }
  }
  
}

# Difference data
diff_data <- data[-data_len,] # take all rows except the final row

for (col in names(data)) {
  for (i in 1:(data_len - 1)) {
    diff_data[i, col] <- data[i, col] / data[i + 1, col]
  }
}

# Categorial data
cat_data <- diff_data
cats <- c("SU", "UP", "NC", "DN", "SD")
get_cat <- function(value, threshold1, threshold2) {
  stopifnot(all(is.numeric(value), is.numeric(threshold1), is.numeric(threshold2), threshold1 > threshold2))
  if (value > (1. + threshold1)) {
    return(cats[1])
  } else if (value > (1. + threshold2)) {
    return(cats[2])
  } else if (value > (1. - threshold2)) {
    return(cats[3])
  } else if (value > (1. - threshold1)) {
    return(cats[4])
  }
  return(cats[5])
}

for (col in c("Open", "High", "Low", "Close")) {
  for (i in 1:(data_len - 1)) {
    cat_data[i, col] <- get_cat(diff_data[i, col], 0.015, 0.005)
  }
}

# Volume
col = "Volume"
for (i in 1:(data_len - 1)) {
  cat_data[i, col] <- get_cat(diff_data[i, col], 0.15, 0.05)
}

# Make values as factor instead of characters
cat_data[,] <- lapply(cat_data[,], as.factor)



####### Grow Decision Tree ###########
# Target: Close
# Features: Open, High, Low, Volume
# Reading: http://scg.sdsu.edu/ctrees_r/ 
require('tree')
custom_control = tree.control(nrow(cat_data), mincut = 5, minsize = 10, mindev = 0.01)
tr <- tree(Close ~ Open + High + Low + Volume, data=cat_data, split="gini", control= custom_control)
plot(tr); text(tr);
summary(tr);

cv.tree(tr, method="misclass");

pr <- prune.misclass(tr);

tr2 <- prune.misclass(tr, k = 0.25);

plot(tr2); text(tr2);
summary(tr2);


fpreds = predict.tree(tr, newdata=cat_data, type="class")
ftable = table(actual=cat_data$Close, fpreds)
ftable

# 
# # Load library for decision tree
# require('rpart')
# require('rpart.plot') # for plotting the decision tree
# 
# # Parameters, TODO: TWEAK THIS
# custom_control = rpart.control(minsplit = 16, minbucket=8, cp = 0.005, maxdepth = 30)
# 
# 
# # Construct tree
# fit <- rpart(Close ~ Open + High + Low + Volume, data=cat_data, method="class", control=custom_control)
# 
# printcp(fit) # display the results
# plotcp(fit) # visualize cross-validation results
# summary(fit) # detailed summary of splits
# 
# # Confusion matrix
# fpreds = predict(fit, newdata=cat_data, type="class")
# ftable = table(actual=cat_data$Close, fpreds)
# ftable
# 
# # plot tree
# prp(fit, type=3, extra=102)
# 
# 
# 
# 
# # Prune tree, TODO: TWEAK THIS TOO
# fit2 <- prune(fit, cp = 0.03)
# 
# # plot tree
# prp(fit2, type=3, extra=102)
# 
# printcp(fit2) # display the results
# plotcp(fit2) # visualize cross-validation results
# summary(fit2) # detailed summary of splits
# 
# # Confusion matrix
# fpreds = predict(fit2, newdata=cat_data, type="class")
# ftable = table(actual=cat_data$Close, fpreds)
# ftable

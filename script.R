# Define random number seed, for constancy across experiments
random_seed <- 42;
# 42, answer to the ultimate question of life, the universe, and everything :P
set.seed(random_seed)


# Load libraries
require('cvTools')# For cross-validation
require('tree')   # Decision tree library
require('e1071')  # Required by 'caret'
require('caret')  # To compute confusion matrix

############### Preprocessing ################

# Open data.csv
raw_data <- read.csv("Data.csv")

# Discard "adj.close" --> Only selects those relevant columns
data <- raw_data[, c("Open", "High", "Low", "Close", "Volume")]

# Discard row where Volume is NA
data <- data[complete.cases(data[, c("Volume")]), ]
data_len <- nrow(data)

# Get the next index at the column which is not NA
next_non_na <- function(data, col, min_idx) {
  stopifnot(is.data.frame(data))
  stopifnot(is.numeric(min_idx))
  
  i = min_idx
  data_len <- nrow(data)
  while (i < data_len + 1) { # Loop it through
    if (!is.na(data[i, col])) {
      return(i)
    }
    i <- i + 1
  }
  return(-1) # not found
}

# Handle missing data
for (col in names(data)) {
  # beginning
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
  
  # end
  # fill in last data
  if (max_idx < data_len) {
    for (i in (max_idx + 1):(data_len)) {
      data[i, col] <- data[max_idx, col]
    }
  }
  
}

# Difference data
diff_data <- data[-data_len, ] # take all rows except the final row

for (col in names(data)) {
  for (i in 1:(data_len - 1)) {
    diff_data[i, col] <- data[i, col] / data[i + 1, col]
  }
}

# Categorial data
cat_data <- diff_data
cats <- c("SU", "UP", "NC", "DN", "SD")
get_cat <- function(value, threshold1, threshold2) {
  # Given a value & its thresholds, what's its category?
  stopifnot(all(
    is.numeric(value),
    is.numeric(threshold1),
    is.numeric(threshold2),
    threshold1 > threshold2
  ))
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

# For "Open", "High", "Low", "Close", thresholds are 0.015 and 0.005
for (col in c("Open", "High", "Low", "Close")) {
  for (i in 1:(data_len - 1)) {
    cat_data[i, col] <- get_cat(diff_data[i, col], 0.015, 0.005)
  }
}

# For "Volume", thresholds are 0.15 and 0.05
col = "Volume"
for (i in 1:(data_len - 1)) {
  cat_data[i, col] <- get_cat(diff_data[i, col], 0.15, 0.05)
}

# Make values as factor instead of characters (required by "tree" library)
cat_data[, ] <- lapply(cat_data[, ], as.factor)

# Shuffle the data
cat_data <- as.data.frame(apply(cat_data, 2, sample))

####### Grow Decision Tree ###########
# Target: Close
# Features: Open, High, Low, Volume
# Reading: http://scg.sdsu.edu/ctrees_r/

# Define pre-pruning conditions
mincuts <- 1:50
n_terminal_nodes_candidates <- 2:10


# Cross validation
nfolds <- 20
folds <- cvFolds(nrow(cat_data), K = nfolds)


for (mincut in mincuts) {
  for (n_terminal_nodes in n_terminal_nodes_candidates) {
    set.seed(random_seed) # Reset seed at each trial of pre-pruning configs
    
    # Init
    accuracy.train.before <- 0.
    accuracy.train.after <- 0.
    accuracy.test.before <- 0.
    accuracy.test.after <- 0.
    
    for (i in 1:nfolds) {
      # Prepare dataset: which ones to use as "train" and "test" (validation)
      val_ind <- folds$subsets[folds$which == i]
      
      cat_data.train <- cat_data[-val_ind,]
      cat_data.test <- cat_data[val_ind,]
      
      # Set the pre-pruning parameters
      custom_control = tree.control(nrow(cat_data.train),
                                    mincut = mincut,
                                    minsize = mincut * 2)
      
      # Build the tree
      tr <-
        tree(
          Close ~ Open + High + Low + Volume,
          data = cat_data.train,
          split = "gini",
          control = custom_control
        )
      
      
      # Test the tree (predict and evaluate with test/validation data)
      fpreds = predict(tr, newdata = cat_data.test, type = "class")
      ftable = table(actual = cat_data.test$Close, fpreds)
      cf = confusionMatrix(ftable)
      
      # Accumulate accuracies before pruning
      accuracy.train.before <-
        accuracy.train.before + (nrow(cat_data.train) - misclass.tree(tr)) / nrow(cat_data.train)
      accuracy.test.before <-
        accuracy.test.before +  cf$overall["Accuracy"]
      
      
      # Post-pruning: Prune the tree
      tr2 <- prune.misclass(tr, best = n_terminal_nodes)
      
      # Test the pruned tree (predict and evaluate with test/validation data)
      fpreds = predict(tr2, newdata = cat_data.test, type = "class")
      ftable = table(actual = cat_data.test$Close, fpreds)
      cf = confusionMatrix(ftable)
      
      # Accumulate accuracies after post-pruning
      accuracy.train.after <-
        accuracy.train.after + (nrow(cat_data.train) - misclass.tree(tr2)) / nrow(cat_data.train)
      accuracy.test.after <-
        accuracy.test.after + cf$overall["Accuracy"]
      
      
    }
    # Make it average accuracy
    accuracy.train.before <- accuracy.train.before  / nfolds
    accuracy.train.after <- accuracy.train.after / nfolds
    accuracy.test.before <- accuracy.test.before / nfolds
    accuracy.test.after <- accuracy.test.after / nfolds
    
    
    # cat("\n\nmincut: ", mincut);
    # cat("\nn_terminal_nodes: ", n_terminal_nodes);
    # cat("\nTrain accuracy before pruning: ", accuracy.train.before);
    # cat("\nTest accuracy before pruning: ", accuracy.test.before);
    #
    # cat("\nTrain accuracy after pruning: ", accuracy.train.after);
    # cat("\nTest accuracy after pruning: ", accuracy.test.after);
    
    
    # This format is more compact and easier to copy-paste & process at Excel :P
    cat(
      "\n",
      paste(
        mincut,
        n_terminal_nodes,
        accuracy.train.before,
        accuracy.test.before,
        accuracy.train.after,
        accuracy.test.after
      )
    )
    
  }
}

##### FINAL TRAIN & TEST #####
# Train-test split
smp_size <- floor(0.80 * nrow(cat_data))


# Reset random seed
set.seed(random_seed)
train_ind <- sample(seq_len(nrow(cat_data)), size = smp_size)

cat_data.final_train <- cat_data[train_ind,]
cat_data.final_test <- cat_data[-train_ind,]

# Pre-pruning configuration
mincut <- 24
n_terminal_nodes <- 4
custom_control = tree.control(nrow(cat_data.final_train),
                              mincut = mincut,
                              minsize = mincut * 2)

# Build tree
tr <-
  tree(
    Close ~ Open + High + Low + Volume,
    data = cat_data.final_train,
    split = "gini",
    control = custom_control
  )


# Plot tree
plot(tr); text(tr);
summary(tr);

# Evaluate the tree (predict and evaluate with test data)
fpreds = predict(tr, newdata = cat_data.final_test, type = "class")
ftable = table(actual = cat_data.final_test$Close, fpreds)
cf = confusionMatrix(ftable)

# Calculate accuracies
accuracy.final_train.before <- (nrow(cat_data.final_train) - misclass.tree(tr)) / nrow(cat_data.final_train)
accuracy.final_test.before <- cf$overall["Accuracy"]


# Post-pruning
tr2 <- prune.misclass(tr, best = n_terminal_nodes)

# Plot tree
plot(tr2); text(tr2);
summary(tr2);

# Evaluate the tree (predict and evaluate with test data)
fpreds2 = predict(tr2, newdata = cat_data.final_test, type = "class")
ftable2 = table(actual = cat_data.final_test$Close, fpreds2)
cf2 = confusionMatrix(ftable2)

# Calculate accuracies
accuracy.final_train.after <- (nrow(cat_data.final_train) - misclass.tree(tr2)) / nrow(cat_data.final_train)
accuracy.final_test.after <- cf2$overall["Accuracy"]


cat("\n\nmincut: ", mincut)
cat("\nn_terminal_nodes: ", n_terminal_nodes)
cat("\nTrain accuracy before pruning: ", accuracy.final_train.before)
cat("\nTest accuracy before pruning: ", accuracy.final_test.before)
cat("\nTest statistics before pruning: ")
cf
cat("\nTrain accuracy after pruning: ", accuracy.final_train.after)
cat("\nTest accuracy after pruning: ", accuracy.final_test.after)
cat("\nTest statistics after pruning: ")
cf2
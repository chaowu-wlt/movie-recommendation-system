library(lubridate)



##########################################################
# Create edx set, validation set (final hold-out test set) R4.0 or later
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")

test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


##########################################################################################################
#
# Data cleansing
#
##########################################################################################################

# Look at edx dataset
edx %>% as_tibble()

# Convert the timestamp in the timestamp column to a date in the edx and validation datasets, and create a new column 'review_date' with the date
edx <- mutate(edx, review_date = as_datetime(timestamp))
validation <- mutate(validation, review_date = as_datetime(timestamp))

# Extract movie release year from the title column in the edx and validation datasets, and create a new column 'release_year' with the year
edx <- mutate(edx, release_year = parse_number(str_extract(title, "\\([0-9]{4}\\)")))
validation <- mutate(validation, release_year = parse_number(str_extract(title, "\\([0-9]{4}\\)")))

# Look at two new columns after wrangling
edx %>% select(review_date, release_year) %>% as_tibble()

# Check for missing values in the edx and validation datasets
sum(is.na(edx))
sum(is.na(validation))



##########################################################################################################
# 
# To train the model and optimize the algorithm parameters without using our test set,
# split the edx set into training set and test set
#
##########################################################################################################

set.seed(755)

# Test set will be 20% of edx set
# Training set will be 80% of edx set
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, 
                                  list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# To make sure we don't include users and movies in the test set that do not appear in the training set, 
# we removed these using the semi_join function
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")



##########################################################################################################
#
# Residual means squared error (RMSE)
# this function computes the residual means squared error for a 
# vector of ratings and their corresponding predictors.
#
##########################################################################################################

RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}



##########################################################################################################
#
# Data exploration and analysis
#
##########################################################################################################

# The average rating of all movies across all users
mean(edx$rating)

# Look at how many zeros were given as rating in the edx dataset
edx %>% filter(rating == 0) %>% summarize(count = n()) %>% pull(count)

# Look at the five most given ratings in order from most to least
edx %>% group_by(rating) %>% summarize (count = n()) %>% top_n(5) %>% arrange(desc(count))

# Are half star ratings less common than whole star ratings
edx %>% 
  group_by(rating) %>%
  summarize(count = n()) %>%
  ggplot(aes(x=rating, y=count)) + geom_line()

# Find out which movie has the greatest number of ratings
edx %>% group_by(movieId, title) %>%
  summarize(count = n()) %>%
  arrange (desc(count))

# Find out the number of unique users that provide ratings, for how many unique movies they provided, and how many unique genres are there
edx %>%
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId),
            n_genres = n_distinct(genres))

# Movies vs. number of ratings
edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  xlab("number of ratings (log scale)") +
  ylab("number of movies") +
  ggtitle("Movies")

# Users vs. number of ratings
edx %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  xlab("number of ratings (log scale)") +
  ylab("number of users") +
  ggtitle("Users")

# Genres vs. number of ratings
edx %>%
  dplyr::count(genres) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  xlab("number of ratings (log scale)") +
  ylab("number of genres") +
  ggtitle("Genres")

# Some movies fall under several genres. Define a category as whatever combination appears in this column.
# Compute and plot the average rating for each category
# But only show the genres with greater than or equal to 30000 ratings
edx %>% group_by(genres) %>%
  summarize(n = n(), avg = mean(rating)) %>%
  filter(n >= 30000) %>% 
  mutate(genres = reorder(genres, avg)) %>%
  ggplot(aes(genres, avg)) + 
  geom_point() +
  theme(axis.text.x=element_text(angle=90,hjust=1))

# Look at review_date column	
# Compute the average rating for each week and plot this average against review date 
edx %>% mutate(review_date = round_date(review_date, unit = "week")) %>%
  group_by(review_date) %>%
  summarize(avg = mean(rating)) %>%
  ggplot(aes(review_date, avg)) +
  geom_point() +
  geom_smooth(method = "loess", span = 0.15, method.args = list(degree=1))

# Review Years vs. number of ratings
edx %>%
  dplyr::count(review_year = year(review_date)) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  xlab("number of ratings (log scale)") +
  ylab("number of years") +
  ggtitle("Review Years")

# Compute and plot the average rating and standard error for each review year 
# But only show the years with greater than or equal to 1000 ratings
edx %>% mutate(review_year = year(review_date)) %>% 
  group_by(review_year) %>% 
  summarize(n=n(), avg=mean(rating),se=sd(rating)/sqrt(n()))%>%
  filter(n >= 1000) %>%
  mutate(review_year = reorder(review_year,avg)) %>%
  ggplot(aes(x=review_year, y=avg, ymin=avg-2*se, ymax=avg+2*se)) + 
  geom_point() + 
  geom_errorbar() + 
  theme(axis.text.x=element_text(angle=45,hjust=1))

# Look at movie release year column
# Plot to see how it relates to number of ratings
edx %>% group_by(movieId) %>%
  summarize(n = n(), year = as.character(first(release_year))) %>%
  qplot(year, n, data = ., geom = "boxplot") +
  coord_trans(y = "sqrt") +
  scale_x_discrete(breaks=seq(min(edx$release_year),max(edx$release_year),4)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Select the top 25 movies with the highest average number of ratings per year
edx %>% 
  filter(release_year >= 1993) %>%
  group_by(movieId) %>%
  summarize(n = n(), years = 2019 - first(release_year),
            rating = mean(rating)) %>%
  mutate(rate = n/years) %>%
  top_n(25, rate) %>%
  arrange(desc(rate))

# Plot to see the trend on average number of ratings per year after a movie released
edx %>% 
  filter(release_year >= 1993) %>%
  group_by(movieId) %>%
  summarize(n = n(), years = 2018 - first(release_year),
            rating = mean(rating)) %>%
  mutate(rate = n/years) %>%
  ggplot(aes(rate, rating)) +
  geom_point() +
  geom_smooth()



##########################################################################################################
#
# Method 1: Just the average
#
##########################################################################################################

# The average rating of all movies across all users
mu_hat <- mean(train_set$rating)

# evaluate the model
naive_rmse <- RMSE(test_set$rating, mu_hat)

# Create a table to store the results that we obtain
rmse_results <- tibble(Method = "Just the average", 
                       Test_on = "test set",
                       Parameter = "NA",
                       Value = "NA",
                       RMSE = naive_rmse)
rmse_results %>% knitr::kable()



##########################################################################################################
#
# Method 2: Movie Effect Model
#
##########################################################################################################

mu <- mean(train_set$rating) 

# Movie effect: compute the least squares estimate
# b_i: the average rating for movie i
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# predict the rating using test set
predicted_ratings <- mu + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

# evaluate the model
model_2_rmse <- RMSE(test_set$rating, predicted_ratings)

# Add this model result to the table
rmse_results <- bind_rows(rmse_results,
                          tibble(Method= "Movie Effect Model",
                                 Test_on = "test set",
                                 Parameter = "NA",
                                 Value = "NA",
                                 RMSE = model_2_rmse))
rmse_results %>% knitr::kable()



##########################################################################################################
#
# Method 3: Movie + User Effects Model
# This model adds user effect to the Movie effect model above.
#
##########################################################################################################

# User effect: by taking the average of the residuals obtained after removing the overall mean and the movie effect from the ratings
# b_u: the average rating for movie i by user u
user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# predict the rating using test set	 
predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

# evaluate the model
model_3_rmse <- RMSE(test_set$rating, predicted_ratings)

# Add this model result to the table
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Movie + User Effects Model",  
                                 Test_on = "test set",
                                 Parameter = "NA",
                                 Value = "NA",
                                 RMSE = model_3_rmse))
rmse_results %>% knitr::kable()



##########################################################################################################
#
# Method 4: Regularized Movie + User Effect Model
# Adding penalty term: parameter lambda, by penalizing large estimates that come from small sample sizes.
#
##########################################################################################################

# Use training set and test set to optimize the parameter lambda
lambdas_model4 <- seq(0, 10, 0.5)

model_4_rmse <- sapply(lambdas_model4, function(l){
  
  mu <- mean(train_set$rating)
  
  # regularized movie effect
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  # regularized user effect
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  # predict the rating using test set
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  # evaluate the model
  return(RMSE(test_set$rating, predicted_ratings))
})

# Find the lambda that give the lowest RMSE
lambda_model4 <- lambdas_model4[which.min(model_4_rmse)]

# Add the lowest RMSE to the result table
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Regularized Movie + User Effect Model",  
                                 Test_on = "test set",
                                 Parameter = "lambda",
                                 Value = as.character(lambda_model4),
                                 RMSE = min(model_4_rmse)))
rmse_results %>% knitr::kable()



##########################################################################################################
#
# Method 5: (Regularized Movie + User) + Genres + Review Year + Movie Release Year Effect Model
# This model is additional to the Regularized Movie + User Effect Model above.
# Adding Genres + Review Year + Movie Release Year effect.
#
##########################################################################################################

# Use training set and test set to optimize the parameter lambda
lambdas_model5 <- seq(0, 10, 0.5)

model_5_rmse <- sapply(lambdas_model5, function(l){
  
  mu <- mean(train_set$rating)
  
  # regularized movie effect
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  # regularized user effect
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  # genre effect:
  # by taking the average of the residuals obtained after removing the overall mean, movie, and user effect from the ratings
  genres_avgs <- train_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = mean(rating - mu - b_i - b_u))
  
  # review year effect: 
  # by taking the average of the residuals obtained after removing the overall mean, movie, user, and genres effect from the ratings
  review_year_avgs <- train_set %>% 
    mutate(review_year = year(review_date)) %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(genres_avgs, by = "genres") %>%
    group_by(review_year) %>%
    summarize(b_t = mean(rating - mu - b_i - b_u - b_g))
  
  # movie release year effect: 
  # by taking the average of the residuals obtained after removing the overall mean, movie, user, genres, and review year effect from the ratings	
  release_year_avgs <- train_set %>%
    mutate(review_year = year(review_date)) %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(genres_avgs, by = "genres") %>%
    left_join(review_year_avgs, by = "review_year") %>%
    group_by(release_year) %>%
    summarize(b_y = mean(rating - mu - b_i - b_u - b_g - b_t))
  
  # predict the rating using test set
  predicted_ratings <- test_set %>% 
    mutate(review_year = year(review_date)) %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(genres_avgs, by = "genres") %>%
    left_join(review_year_avgs, by = "review_year") %>%
    left_join(release_year_avgs, by = "release_year") %>%
    mutate(pred = mu + b_i + b_u + b_g + b_t + b_y) %>%
    pull(pred)
  
  # evaluate the model
  return(RMSE(test_set$rating, predicted_ratings))
})

# Find the lambda that give the lowest RMSE
lambda_model5 <- lambdas_model5[which.min(model_5_rmse)]

# Add the lowest RMSE to the result table
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="(Regularized Movie + User) + Genres + Review Year + Movie Release Year Effect Model",  
                                 Test_on = "test set",
                                 Parameter = "lambda",
                                 Value = as.character(lambda_model5),
                                 RMSE = min(model_5_rmse)))
rmse_results %>% knitr::kable()



##########################################################################################################
#
# Analysis on regularization on Genres effect
# Build on Method 5: (Regularized Movie + User) + Genres + Review Year + Movie Release Year effect Model
# Adding penalty term: parameter lambda
#
##########################################################################################################

# start with a fixed value
lambda <- lambda_model5
mu <- mean(train_set$rating)

# regularized movie effect
b_i <- train_set %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

# regularized user effect
b_u <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

# genres average effect
genres_avgs <- train_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

# regularized genres effect	
genres_reg_avgs <- train_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+lambda), n_i = n()) 

# review year average effect
review_year_avgs <- train_set %>% 
  mutate(review_year = year(review_date)) %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(genres_reg_avgs, by = "genres") %>%
  group_by(review_year) %>%
  summarize(b_t = mean(rating - mu - b_i - b_u - b_g))

# movie release year average effect 
release_year_avgs <- train_set %>%
  mutate(review_year = year(review_date)) %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(genres_reg_avgs, by = "genres") %>%
  left_join(review_year_avgs, by = "review_year") %>%
  group_by(release_year) %>%
  summarize(b_y = mean(rating - mu - b_i - b_u - b_g - b_t))

# plot the regularization effect using genres average results and regularized effect result
tibble(original = genres_avgs$b_g, 
       regularlized = genres_reg_avgs$b_g, 
       n = genres_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)

# predict ratings using regularized genres effect
reg_genres_predicted_ratings <- test_set %>% 
  mutate(review_year = year(review_date)) %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(genres_reg_avgs, by = "genres") %>%
  left_join(review_year_avgs, by = "review_year") %>%
  left_join(release_year_avgs, by = "release_year") %>%
  mutate(pred = mu + b_i + b_u + b_g + b_t + b_y) %>%
  pull(pred)

# obtain the result
reg_genres_rmse <- RMSE(test_set$rating, reg_genres_predicted_ratings)

# show results to compare genres effect without and with regularization
reg_rmse_results <- tibble(Method = "Genres average effect", 
                           RMSE = min(model_5_rmse))
reg_rmse_results <- bind_rows(reg_rmse_results,
                              tibble(Method="Regularized Genres effect",  
                                     RMSE = reg_genres_rmse))
reg_rmse_results %>% knitr::kable()



##########################################################################################################
#
# Analysis on regularization on Review Year effect
# Build on Method 5: (Regularized Movie + User) + Genres + Review Year + Movie Release Year effect Model
# Adding penalty term: parameter lambda
#
##########################################################################################################

# start with a fixed value
lambda <- lambda_model5
mu <- mean(train_set$rating)

# regularized movie effect
b_i <- train_set %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

# regularized user effect
b_u <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

# genres average effect
genres_avgs <- train_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

# review year average effect
review_year_avgs <- train_set %>% 
  mutate(review_year = year(review_date)) %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(genres_avgs, by = "genres") %>%
  group_by(review_year) %>%
  summarize(b_t = mean(rating - mu - b_i - b_u - b_g))

# regularized review year effect
review_year_reg_avgs <- train_set %>% 
  mutate(review_year = year(review_date)) %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(genres_avgs, by = "genres") %>%
  group_by(review_year) %>%
  summarize(b_t = sum(rating - b_i - b_u - b_g - mu)/(n()+lambda), n_i = n())

# movie release year average effect 
release_year_avgs <- train_set %>%
  mutate(review_year = year(review_date)) %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(genres_avgs, by = "genres") %>%
  left_join(review_year_reg_avgs, by = "review_year") %>%
  group_by(release_year) %>%
  summarize(b_y = mean(rating - mu - b_i - b_u - b_g - b_t))

# plot the regularization effect using review year average results and regularized effect result
tibble(original = review_year_avgs$b_t, 
       regularlized = review_year_reg_avgs$b_t, 
       n = review_year_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=n)) + 
  geom_point(shape=1, alpha=0.5)

# predict ratings using regularized review year effect
reg_review_year_predicted_ratings <- test_set %>% 
  mutate(review_year = year(review_date)) %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(genres_avgs, by = "genres") %>%
  left_join(review_year_reg_avgs, by = "review_year") %>%
  left_join(release_year_avgs, by = "release_year") %>%
  mutate(pred = mu + b_i + b_u + b_g + b_t + b_y) %>%
  pull(pred)

# obtain the result
reg_review_year_rmse <- RMSE(test_set$rating, reg_review_year_predicted_ratings)

# show results to compare review year effect without and with regularization
reg_rmse_results <- tibble(Method = "Review Year average effect", 
                           RMSE = min(model_5_rmse))
reg_rmse_results <- bind_rows(reg_rmse_results,
                              tibble(Method="Regularized Review Year effect",  
                                     RMSE = reg_review_year_rmse))
reg_rmse_results %>% knitr::kable()



##########################################################################################################
#
# Analysis on regularization on Movie Release Year effect
# Build on Method 5: (Regularized Movie + User) + Genres + Review Year + Movie Release Year effect Model
# Adding penalty term: parameter lambda
#
##########################################################################################################

# start with a fixed value
lambda <- lambda_model5
mu <- mean(train_set$rating)

# regularized movie effect
b_i <- train_set %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

# regularized user effect
b_u <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

# genres average effect
genres_avgs <- train_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = mean(rating - mu - b_i - b_u))

# review year average effect
review_year_avgs <- train_set %>% 
  mutate(review_year = year(review_date)) %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(genres_avgs, by = "genres") %>%
  group_by(review_year) %>%
  summarize(b_t = mean(rating - mu - b_i - b_u - b_g))

# movie release year average effect 
release_year_avgs <- train_set %>%
  mutate(review_year = year(review_date)) %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(genres_avgs, by = "genres") %>%
  left_join(review_year_avgs, by = "review_year") %>%
  group_by(release_year) %>%
  summarize(b_y = mean(rating - mu - b_i - b_u - b_g - b_t))

# regularized movie release year effect
release_year_reg_avgs <- train_set %>%
  mutate(review_year = year(review_date)) %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(genres_avgs, by = "genres") %>%
  left_join(review_year_avgs, by = "review_year") %>%
  group_by(release_year) %>%
  summarize(b_y = sum(rating - mu - b_i - b_u - b_g - b_t)/(n()+lambda), n_i = n())

# plot the regularization effect using release year average results and regularized result
tibble(original = release_year_avgs$b_y, 
       regularlized = release_year_reg_avgs$b_y, 
       n = release_year_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=n)) + 
  geom_point(shape=1, alpha=0.5)

# predict ratings using regularized release year effect
reg_release_year_predicted_ratings <- test_set %>% 
  mutate(review_year = year(review_date)) %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(genres_avgs, by = "genres") %>%
  left_join(review_year_avgs, by = "review_year") %>%
  left_join(release_year_reg_avgs, by = "release_year") %>%
  mutate(pred = mu + b_i + b_u + b_g + b_t + b_y) %>%
  pull(pred)

# obtain the result
reg_release_year_rmse <- RMSE(test_set$rating, reg_release_year_predicted_ratings)

# show results to compare release year effect without and with regularization
reg_rmse_results <- tibble(Method = "Release Year average effect", 
                           RMSE = min(model_5_rmse))
reg_rmse_results <- bind_rows(reg_rmse_results,
                              tibble(Method="Regularized Release Year effect",  
                                     RMSE = reg_release_year_rmse))
reg_rmse_results %>% knitr::kable()



##########################################################################################################
#
# Method 6: Regularized Movie + User + Genres + Review Year + Movie Release Year Effect Model
# This is additional to method 5 above.
# Adding penalty term: parameter lambda, by penalizing large estimates that come from small sample sizes.
# This model adds regularization to Genres, Review Year, and Movie Release Year effect.
#
##########################################################################################################

# Use training set and test set to optimize the parameter lambda that will give lowest RMSE
lambdas_model6 <- seq(0, 10, 0.5)

model_6_rmse <- sapply(lambdas_model6, function(l){
  mu <- mean(train_set$rating)
  
  # regularized movie effect
  b_i <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  # regularized user effect
  b_u <- train_set %>% 
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  # regularized genres effect
  b_g <- train_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    group_by(genres) %>%
    summarize(b_g = sum(rating - mu - b_i - b_u)/(n()+l))
  
  # regularized review year effect
  b_t <- train_set %>% 
    mutate(review_year = year(review_date)) %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    group_by(review_year) %>%
    summarize(b_t = sum(rating - b_i - b_u - b_g - mu)/(n()+l))
  
  # regularized movie release year effect
  b_y <- train_set %>%
    mutate(review_year = year(review_date)) %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_t, by = "review_year") %>%
    group_by(release_year) %>%
    summarize(b_y = sum(rating - mu - b_i - b_u - b_g - b_t)/(n()+l))
  
  # predict the rating using test set
  predicted_ratings <- test_set %>% 
    mutate(review_year = year(review_date)) %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_g, by = "genres") %>%
    left_join(b_t, by = "review_year") %>%
    left_join(b_y, by = "release_year") %>%
    mutate(pred = mu + b_i + b_u + b_g + b_t + b_y) %>%
    pull(pred)
  
  # evaluate the model
  return(RMSE(test_set$rating, predicted_ratings))
})

# Find the lambda that give the lowest RMSE
lambda_model6 <- lambdas_model6[which.min(model_6_rmse)]

# Add the lowest RMSE to the result table
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Regularized Movie + User + Genres + Review Year + Movie Release Year Effect Model",  
                                 Test_on = "test set",
                                 Parameter = "lambda",
                                 Value = as.character(lambda_model6),
                                 RMSE = min(model_6_rmse)))
rmse_results %>% knitr::kable()



##########################################################################################################
#
# Predict the rating using the final model: 
# Regularized Movie + User + Genres + Review Year + Movie Release Year Effect Model 
# Using lambda lambda_model6 that gives the lowest RMSE in the model of Method 6
# Dataset: whole edx dataset for training, validation for testing
#
##########################################################################################################

mu <- mean(edx$rating)

# regularized movie effect
b_i <- edx %>% 
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda_model6))

# regularized user effect
b_u <- edx %>% 
  left_join(b_i, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda_model6))

# regularized genre effect
b_g <- edx %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - b_i - b_u - mu)/(n()+lambda_model6))

# regularized review year effect
b_t <- edx %>% 
  mutate(review_year = year(review_date)) %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  group_by(review_year) %>%
  summarize(b_t = sum(rating - b_i - b_u - b_g - mu)/(n()+lambda_model6))

# regularized movie release year effect
b_y <- edx %>%
  mutate(review_year = year(review_date)) %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_t, by = "review_year") %>%
  group_by(release_year) %>%
  summarize(b_y = sum(rating - mu - b_i - b_u - b_g - b_t)/(n()+lambda_model6))

# predict the rating using validation set
predicted_ratings <- validation %>% 
  mutate(review_year = year(review_date)) %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_g, by = "genres") %>%
  left_join(b_t, by = "review_year") %>%
  left_join(b_y, by = "release_year") %>%
  mutate(pred = mu + b_i + b_u + b_g + b_t + b_y) %>%
  pull(pred)

# evaluate the model
final_model_rmse <- RMSE(validation$rating, predicted_ratings)

# Add this model result to the table
rmse_results <- bind_rows(rmse_results,
                          tibble(Method="Final model: Regularized Movie + User + Genres + Review Year + Movie Release Year Effect Model",  
                                 Test_on = "validation set",
                                 Parameter = "lambda",
                                 Value = as.character(lambda_model6),
                                 RMSE = final_model_rmse))
rmse_results %>% knitr::kable()


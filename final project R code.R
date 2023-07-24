population_mean <- 22
population_sd <- 1.5
#population size
N<- 500
#sample size 
n <- 500
errors <- numeric(N)

par(mfrow=c(3,1))

for (s in 1:N) {
  
  sample_means <- numeric(n)
  
  # Draw random samples and compute means
  for (i in 1:n) {
    sample <- rnorm(s, population_mean, population_sd)
    sample_means[i] <- mean(sample)
  }
  
  # Compute RMSE and store it to errors
  errors[s] <- sqrt(mean((sample_means - population_mean)^2))
}

# Plot errors against sample size
plot(1:N, errors, type="l",xlab="sample size", 
     ylab="RMSE for normal distribution")

error_df <- data.frame(sample_size = 1:N, RMSE = errors)
head(error_df)



##2: exponential distribution
population_distribution <- rexp

# population size
N <- 500

# sample size
n <- 500

# Population mean for exponential distribution

lambda <- 1/population_mean

# Initialize vector to store errors for exponential distribution
errors_exp <- numeric(N)

# Loop over different sample sizes
for (s in 1:n) {
  
  # Initialize vector to store sample means for exponential distribution
  sample_means_exp <- numeric(n)
  
  # Draw random samples and compute means for exponential distribution
  for (i in 1:n) {
    sample_exp <- population_distribution(s,rate = lambda)
    sample_means_exp[i] <- mean(sample_exp)
  }
  
  
  # Compute RMSE and store it to errors for exponential distribution
  errors_exp[s] <- sqrt(mean((sample_means_exp - population_mean)^2))
}

# Plot errors for exponential distribution against sample size
plot(1:N, errors_exp, type = "l", xlab = "sample size", ylab = "RMSE for exponential distribution")

# Create data frame of RMSE values for exponential distribution
error_exp_df <- data.frame(sample_size = 1:N, RMSE = errors_exp)
head(error_exp_df)



##3 
# Set population parameters
population_mean <- 22
population_sd <- 1.5

# Set population distribution to be rgama
pop_distribution <- rgamma

# Set population size
N <- 500

# Set sample size
n <- 500

# Initialize vector to store errors for gamma distribution
errors_gamma <- numeric(N)

# Set parameters for gamma distribution 
shape <- (population_mean^2)/(population_sd^2)
rate <- shape / population_mean

# Loop over different sample sizes
for (s in 1:N) {
  
  # Initialize vector to store sample means for gamma distribution
  sample_means_gamma <- numeric(n)
  
  # Draw random samples and compute means for gamma distribution
  for (i in 1:n) {
    sample_gamma <- pop_distribution(s, shape , rate )
    sample_means_gamma[i] <- mean(sample_gamma)
  }
  
  # Compute RMSE and store it to errors for gamma distribution
  errors_gamma[s] <- sqrt(mean((sample_means_gamma - population_mean)^2))
}

# Plot errors for gamma distribution against sample size
plot(1:N, errors_gamma, type = "l", xlab = "sample size", ylab = "RMSE for gamma distribution")

# Create data frame of RMSE values for gamma distribution
error_gamma_df <- data.frame(n = 1:N, RMSE = errors_gamma)
head(error_gamma_df)



#-----------------------------------------------------------------------------#
# Calculate the difference between each RMSE data points
# ----------------------------------------------------------------------------#

# Normal Distribution #
rmse_diff <- numeric(nrow(error_df))
for (i in 2:nrow(error_df)) {
  rmse_diff[i] <- error_df$RMSE[i] - error_df$RMSE[i-1]
}
error_df$rmse_diff <- rmse_diff
error_df$rmse_diff <- round(error_df$rmse_diff, 3)
#--------------------------------------------------------------------------------
# Exponential Distrubution #
rmse_diff_exp <- numeric(nrow(error_exp_df))
for (i in 2:nrow(error_exp_df)) {
  rmse_diff_exp[i] <- error_exp_df$RMSE[i] - error_exp_df$RMSE[i-1]
}
error_exp_df$rmse_diff_exp <- rmse_diff_exp
error_exp_df$rmse_diff_exp <- round(error_exp_df$rmse_diff_exp, 3)

#--------------------------------------------------------------------------------
# Gamma Distribution #
RMSE_diff_gamma <- numeric(nrow(error_gamma_df))
for (i in 2:nrow(error_gamma_df)) {
  RMSE_diff_gamma[i] <- error_gamma_df[i, "RMSE"] - error_gamma_df[i-1, "RMSE"]
}
error_gamma_df$RMSE_diff_gamma <- RMSE_diff_gamma
error_gamma_df$RMSE_diff_gamma <- round(RMSE_diff_gamma, 3)


# ----------------------------------------------------------------------------#
# Cost-benefit analysis 
# ----------------------------------------------------------------------------#

# Normal Distribution #
# Set up variables
sample_size <- 110
cost_per_person <- 10
total_cost <- sample_size * cost_per_person
mean <- 22
se <- 1.5/sqrt(sample_size)
print (se)
# Calculate the standard error of the mean
sem <- se / sqrt(sample_size)

# Calculate the margin of error (using a 95% confidence level)
z_score <- qnorm(0.975) # 2-tailed test
moe <- z_score * sem

# Calculate the confidence interval
lower_ci <- mean - moe
upper_ci <- mean + moe

# Calculate the expected net benefit
benefit <- (upper_ci - lower_ci) / 2
net_benefit <- benefit - total_cost

# Print the results
cat("Total cost: $", total_cost, "\n")
cat("95% confidence interval: [", lower_ci, ", ", upper_ci, "]\n")
cat("Expected net benefit: $", net_benefit, "\n")
#--------------------------------------------------------------------------------
# Exponential Distribution
# Set up variables
sample_size_exp <- 40
cost_per_person <- 10
total_cost_exp <- sample_size_exp * cost_per_person
population_sd <- 1.5
mean <- 22

# Calculate the standard error of the mean
se_exp <- population_sd / sqrt(sample_size_exp)

# Calculate the margin of error (using a 95% confidence level)
z_score <- qnorm(0.975) # 2-tailed test
moe_exp <- z_score * se_exp

# Calculate the confidence interval
lower_ci_exp <- mean - moe_exp
upper_ci_exp <- mean + moe_exp

# Calculate the expected net benefit
benefit_exp <- (upper_ci_exp - lower_ci_exp) / 2
net_benefit_exp <- benefit_exp - total_cost_exp

# Print the results
cat("Total cost: $", total_cost_exp, "\n")
cat("95% confidence interval: [", lower_ci_exp, ", ", upper_ci_exp, "]\n")
cat("Expected net benefit: $", net_benefit_exp, "\n")
#--------------------------------------------------------------------------------
# Gamma Distribution
sample_size_gamma <- 27
cost_per_person <- 10
total_cost_gamma <- sample_size_gamma * cost_per_person
population_sd <- 1.5
mean <- 2227

# Calculate the standard error of the mean
se_gamma <- population_sd / sqrt(sample_size_gamma)

# Calculate the margin of error (using a 95% confidence level)
z_score <- qnorm(0.975) # 2-tailed test
moe_gamma <- z_score * se_gamma

# Calculate the confidence interval
lower_ci_gamma <- mean - moe_gamma
upper_ci_gamma <- mean + moe_gamma

# Calculate the expected net benefit
benefit_gamma <- (upper_ci_gamma - lower_ci_gamma) / 2
net_benefit_gamma <- benefit_gamma - total_cost_gamma

# Print the results
cat("Total cost: $", total_cost_gamma, "\n")
cat("95% confidence interval: [", lower_ci_gamma, ", ", upper_ci_gamma, "]\n")
cat("Expected net benefit: $", net_benefit_gamma, "\n")

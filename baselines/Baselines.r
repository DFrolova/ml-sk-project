#Baseline models: SNaive, Theta, ARIMA
#Metrics: MAPE, sMAPE, ND
#Datasets: TOURISM, M3, M4, FRED, TRAFFIC, ELECTRICITY

#Some datasets are really large, script run takes a lot of time (~2-3 days of pure time for all datasets). The most
#straightforward way to check this script is to define all needed functions (till baselines_forecast), specify path to
#TOURISM dataset and run the part of the script for TOURISM dataset (this dataset is quite small and so is running time).
#Obtained numbers will be the same as in the project report. For other parts of the script the procedure may be not that
#straightforward.

# these libraries should be installed in order to run the script
library(parallel)
library(Metrics)
library(forecast)
library(forecTheta)


# baseline models 
Naive <- function(data, leng, h){
  len = unlist(leng)
  n = length(len)
  
  i <- 1
  y_pred <- matrix(0, h, n)
  for (val in len){
    y_pred[,i] <- data[val,i]
    i <- i + 1
  }
  
  return(y_pred)
}

SNaive <- function(data, leng, h, m, fl="Seasonal"){
  len <- unlist(leng)
  n <- length(len)
  
  y_pred <- matrix(0, h, n)
  i <- 1
  for (val in len){
    if (fl != "Seasonal"){
      y_pred[,i] <- data[val,i]
    } else{
      for (h_ in 1:h){
        h_m <- (h_-1) %% m + 1
        j <- val-m+h_m
        y_pred[h_,i] <- data[j,i]
      }
    }
    i <- i + 1
  }
  
  return(y_pred)
}

theta <- function(data, leng, h, m){
  len <- unlist(leng)
  n <- length(len)
  
  y_pred <- matrix(0,h,n)
  for (i in 1:n){
    y <- data[1:len[i],i]
    if (len[i] > 5000){
      y <- data[(len[i]-4000):len[i],i]
    }
    if (m == 1){
      y <- ts(y)
      #y_pred[,i] <- dotm(y, h, s=NULL)$mean
      y_pred[,i] <- thetaf(y, h)$mean
    } else{
      n_end <- len[i] %/% m
      m_end <- len[i] %% m
      if (m_end == 0){
        m_end = m
      } else{
        n_end <- n_end + 1
      }
      y <- ts(y, start=c(1,1), end=c(n_end,m_end), frequency=m)
      
      #if (len[i] < (2*m)){
        
      #} else{
        #y_pred[,i] <- dotm(y, h, s="additive", par_ini=c(0.5,0.5,2))$mean
      #}
      y_pred[,i] <- thetaf(y, h)$mean
    }
  }
  
  return (y_pred)
}

arima_forecast <- function(data, leng, h, m){
  len = unlist(leng)
  n = length(len)
  
  y_pred = matrix(0,h,n)
  for (i in 1:n){
    y <- data[1:len[i],i]
    if (len[i] > 5000){
      y <- data[(len[i]-4000):len[i],i]
    }
    if (m == 1){
      y <- ts(y)
      model <- auto.arima(y, seasonal=FALSE, ic=c("aic"), stepwise=TRUE, approximation=TRUE)
    } else{
      n_end = len[i] %/% m
      m_end = len[i] %% m
      if (m_end == 0){
        m_end = m
      } else{
        n_end <- n_end + 1
      }
      y <- ts(y, start=c(1,1), end=c(n_end,m_end), frequency=m)
      model <- auto.arima(y, seasonal=TRUE, ic=c("aic"), nmodels=20, stepwise=TRUE, approximation=TRUE)
    }
    y_pred[,i] <- forecast(model, h)$mean
  }
  
  return(y_pred)
}

# used metrics
MAPE_average <- function(y_test, y_pred){
  n = dim(y_pred)[2]
  res <- 0
  for (i in 1:n){
    res <- res + mape(y_test[,i], y_pred[,i])
  }
  res <- res / n * 100
  
  return(res)
}

sMAPE_average <- function(y_test, y_pred){
  n = dim(y_pred)[2]
  res <- 0
  for (i in 1:n){
    if (!is.na(smape(y_test[,i], y_pred[,i]))){
      res <- res + smape(y_test[,i], y_pred[,i])
    }
  }
  res <- res / n * 100
  
  return(res)
}

ND <- function(y_test, y_pred){
  error <- abs(sum(y_test - y_pred))
  res <- error / sum(abs(y_test))
  
  return (res)
}

# main function
baselines_forecast <- function(y_train, y_test, y_length, y_h, y_m, metric, fl="Seasonal"){
  
  y_SNaive_pred <- SNaive(y_train, y_length, y_h, y_m, fl=fl)
  y_theta_pred <- theta(y_train, y_length, y_h, y_m)
  y_arima_pred <- arima_forecast(y_train, y_length, y_h, y_m)
  
  if (metric == "MAPE"){
    SNaive_mape <- MAPE_average(y_test, y_SNaive_pred)
    theta_mape <- MAPE_average(y_test, y_theta_pred)
    arima_mape <- MAPE_average(y_test, y_arima_pred)
    
    res <- c(SNaive_mape, theta_mape, arima_mape)
  } else if (metric == "sMAPE"){
    SNaive_smape <- sMAPE_average(y_test, y_SNaive_pred)
    theta_smape <- sMAPE_average(y_test, y_theta_pred)
    arima_smape <- sMAPE_average(y_test, y_arima_pred)
    
    res <- c(SNaive_smape, theta_smape, arima_smape)
  } else if (metric == "ND"){
    SNaive_nd <- ND(y_test, y_SNaive_pred)
    theta_nd <- ND(y_test, y_theta_pred)
    arima_nd <- ND(y_test, y_arima_pred)
    
    res <- c(SNaive_nd, theta_nd, arima_nd)
    #res <- list("snaive" = y_SNaive_pred, "theta" = y_theta_pred, "arima" = y_arima_pred)
  }
  
  return(res)
}

#path to the folder where datasets are. For tourism one need to specify only the path to the 'tourism' folder whereas
#for other datasets path should be specified manually in read.csv functions
path <- "~/Scripts/Machine Learning/Project/R/"

#tourism data preparation
year_train <- read.csv(paste(path, "tourism/yearly-train.csv", sep=""))
year_test <- read.csv(paste(path, "tourism/yearly-test.csv", sep=""))
year_length = lapply(year_train, function(s){s[1]})
year_h = 4
year_m = 1
year_train = year_train[3:(max(unlist(year_length))+2),]
year_test = year_test[3:(year_h+2),]

quarter_train <- read.csv(paste(path, "tourism/quarterly-train.csv", sep=""))
quarter_test <- read.csv(paste(path, "tourism/quarterly-test.csv", sep=""))
quarter_length = lapply(quarter_train, function(s){s[1]})
quarter_h = 8
quarter_m = 4
quarter_start <- quarter_train[3,]
quarter_train = quarter_train[4:(max(unlist(quarter_length))+3),]
quarter_test = quarter_test[4:(quarter_h+3),]

month_train <- read.csv(paste(path, "tourism/monthly-train.csv", sep=""))
month_test <- read.csv(paste(path, "tourism/monthly-test.csv", sep=""))
month_length = lapply(month_train, function(s){s[1]})
month_h = 24
month_m = 12
month_start <- month_train[3,]
month_train = month_train[4:(max(unlist(month_length))+3),]
month_test = month_test[4:(month_h+3),]

# prediction and metric evaluation
year_mape <- baselines_forecast(year_train, year_test, year_length, year_h, year_m, "MAPE", fl="Non Seasonal")
quarter_mape <- baselines_forecast(quarter_train, quarter_test, quarter_length, quarter_h, quarter_m, "MAPE")
month_mape <- baselines_forecast(month_train, month_test, month_length, month_h, month_m, "MAPE")

#final results
tourism_data <- data.frame("Year" = year_mape, "Quarter" = quarter_mape, "Month" = month_mape, row.names=c("SNaive", "Theta", "ARIMA"))
tourism_data


#M3 data preparation
year_data <- read.csv("~/Scripts/Machine Learning/Project/R/M3/M3Year.csv")
quarter_data <- read.csv("~/Scripts/Machine Learning/Project/R/M3/M3Quart.csv")
month_data <- read.csv("~/Scripts/Machine Learning/Project/R/M3/M3Month.csv")
other_data <- read.csv("~/Scripts/Machine Learning/Project/R/M3/M3Other.csv")

M3_preprocess <- function(yy, h){
  n <- dim(yy)
  y_length <- yy[2]
  y <- t(yy[,7:n[2]])
  n <- dim(y)
  y_length <- y_length - h
  y_train <- matrix(0,n[1]-h,n[2])
  y_test <- matrix(0,h,n[2])
  for (j in 1:n[2]){
    l <- y_length[j,1]
    y_train[1:l,j] <- y[1:l,j]
    y_test[,j] <- y[(l+1):(l+h),j]
  }
  
  res = list("train" = y_train, "test" = y_test, "len" = y_length)
  
  return (res)
}

year_h <- 6
year_m <- 1

data <- M3_preprocess(year_data, year_h)
year_train <- data$train
year_test <- data$test
year_length <- data$len

#prediction and metric evaluation
year_mape <- baselines_forecast(year_train, year_test, year_length, year_h, year_m, "sMAPE")
year_mape

quarter_h <- 8
quarter_m <- 4

data <- M3_preprocess(quarter_data, quarter_h)
quarter_train <- data$train
quarter_test <- data$test
quarter_length <- data$len

quarter_naive_pred <- SNaive(quarter_train, quarter_length, quarter_h, quarter_m, fl="Seasonal")
quarter_naive_smape <- sMAPE_average(quarter_test, quarter_naive_pred)
quarter_naive_smape

quarter_mape <- baselines_forecast(quarter_train, quarter_test, quarter_length, quarter_h, quarter_m, "sMAPE")


month_h <- 18
month_m <- 12

data <- M3_preprocess(month_data, month_h)
month_train <- data$train
month_test <- data$test
month_length <- data$len

month_mape <- baselines_forecast(month_train, month_test, month_length, month_h, month_m, "sMAPE")


other_h <- 8
other_m <- 1

data <- M3_preprocess(other_data, other_h)
other_train <- data$train
other_test <- data$test
other_length <- data$len

other_naive_pred <- SNaive(other_train, other_length, other_h, other_m, fl="Non Seasonal")
other_naive_smape <- sMAPE_average(other_test, other_naive_pred)
other_naive_smape

other_mape <- baselines_forecast(other_train, other_test, other_length, other_h, other_m, "sMAPE")
other_mape


#final results
M3_data <- data.frame("Year" = year_mape, "Quarter" = quarter_mape, "Month" = month_mape, "Other" = other_mape, row.names=c("SNaive", "Theta", "ARIMA"))
M3_data


#M4 data preparation
year_train <- read.csv("~/Scripts/Machine Learning/Project/R/M4/train/Yearly-train.csv")
quarter_train <- read.csv("~/Scripts/Machine Learning/Project/R/M4/train/Quarterly-train.csv")
month_train <- read.csv("~/Scripts/Machine Learning/Project/R/M4/train/Monthly-train.csv")
week_train <- read.csv("~/Scripts/Machine Learning/Project/R/M4/train/Weekly-train.csv")
day_train <- read.csv("~/Scripts/Machine Learning/Project/R/M4/train/Daily-train.csv")
hour_train <- read.csv("~/Scripts/Machine Learning/Project/R/M4/train/Hourly-train.csv")

year_test <- read.csv("~/Scripts/Machine Learning/Project/R/M4/test/Yearly-test.csv")
quarter_test <- read.csv("~/Scripts/Machine Learning/Project/R/M4/test/Quarterly-test.csv")
month_test <- read.csv("~/Scripts/Machine Learning/Project/R/M4/test/Monthly-test.csv")
week_test <- read.csv("~/Scripts/Machine Learning/Project/R/M4/test/Weekly-test.csv")
day_test <- read.csv("~/Scripts/Machine Learning/Project/R/M4/test/Daily-test.csv")
hour_test <- read.csv("~/Scripts/Machine Learning/Project/R/M4/test/Hourly-test.csv")

year_h <- 6
quarter_h <- 8
month_h <- 18
week_h <- 13
day_h <- 14
hour_h <- 48

year_m <- 1
quarter_m <- 4
month_m <- 12
week_m <- 7
day_m <- 1
hour_m <- 24

n_year_train = dim(year_train)
year_train_ <- t(year_train[,2:n_year_train[2]])
year_test_ <- t(year_test[,2:(year_h+1)])

n_quarter_train = dim(quarter_train)
quarter_train_ <- t(quarter_train[,2:n_quarter_train[2]])
quarter_test_ <- t(quarter_test[,2:(quarter_h+1)])

n_month_train = dim(month_train)
month_train_ <- t(month_train[,2:n_month_train[2]])
month_test_ <- t(month_test[,2:(month_h+1)])

n_week_train = dim(week_train)
week_train_ <- t(week_train[,2:n_week_train[2]])
week_test_ <- t(week_test[,2:(week_h+1)])

n_day_train = dim(day_train)
day_train_ <- t(day_train[,2:n_day_train[2]])
day_test_ <- t(day_test[,2:(day_h+1)])

n_hour_train <- dim(hour_train)
hour_train_ <- t(hour_train[,2:n_hour_train[2]])
hour_test_ <- t(hour_test[,2:(hour_h+1)])


year_length <- apply(year_train_, 2, function(x) length(which(!is.na(x))))
quarter_length <- apply(quarter_train_, 2, function(x) length(which(!is.na(x))))
month_length <- apply(month_train_, 2, function(x) length(which(!is.na(x))))
week_length <- apply(week_train_, 2, function(x) length(which(!is.na(x))))
day_length <- apply(day_train_, 2, function(x) length(which(!is.na(x))))
hour_length <- apply(hour_train_, 2, function(x) length(which(!is.na(x))))


#predictions and metric evaluation
week_smape <- baselines_forecast(week_train_, week_test_, week_length, week_h, week_m, "sMAPE")
week_smape
hour_smape <- baselines_forecast(hour_train_, hour_test_, hour_length, hour_h, hour_m, "sMAPE")
hour_smape
day_smape <- baselines_forecast(day_train_, day_test_, day_length, day_h, day_m, "sMAPE")
day_smape
year_smape <- baselines_forecast(year_train_, year_test_, year_length, year_h, year_m, "sMAPE")
year_smape
quarter_smape <- baselines_forecast(quarter_train_, quarter_test_, quarter_length, quarter_h, quarter_m, "sMAPE")
quarter_smape
month_smape <- baselines_forecast(month_train_, month_test_, month_length, month_h, month_m, "sMAPE")
month_smape


#final results
M4_data <- data.frame("Year" = year_smape, "Quarter" = quarter_smape, "Month" = month_smape, "Week" = week_smape, "Day" = day_smape, "Hour" = hour_smape, row.names=c("SNaive", "Theta", "ARIMA"))
M4_data


#traffic and electricity data preparation
traffic_data <- read.csv("~/Scripts/Machine Learning/Project/R/traffic/traffic.csv")
traffic_length <- apply(traffic_data, 2, function(x) length(which(!is.na(x))))

electricity_data <- read.csv("~/Scripts/Machine Learning/Project/R/Electricity/electricity.csv")
electricity_length <- apply(electricity_data, 2, function(x) length(which(!is.na(x))))

traffic_preprocess <- function(yy, h){
  n <- dim(yy)
  y <- yy[,2:n[2]]
  n <- dim(y)
  y_length <- apply(y, 2, function(x) length(which(!is.na(x))))
  y_length <- unlist(y_length)
  y_length <- y_length - h
  y_train <- matrix(0,n[1]-h,n[2])
  y_test <- matrix(0,h,n[2])
  for (j in 1:n[2]){
    l <- y_length[j]
    y_train[1:l,j] <- y[1:l,j]
    y_test[,j] <- y[(l+1):(l+h),j]
  }
  
  res = list("train" = y_train, "test" = y_test, "len" = y_length)
  
  return (res)
}

#prediction and metric evaluation
traffic_h <- 24*7
data <- traffic_preprocess(traffic_data, traffic_h)
traffic_train <- data$train
traffic_test <- data$test
traffic_length <- data$len

traffic_nd <- baselines_forecast(traffic_train, traffic_test, traffic_length, traffic_h, 1, "ND", fl="Non Seasonal")
traffic_nd

electricity_h <- 24*7
data <- traffic_preprocess(electricity_data, electricity_h)
electricity_train <- data$train
electricity_test <- data$test
electricity_length <- data$len

electricity_nd <- baselines_forecast(electricity_train, electricity_test, electricity_length, electricity_h, 1, "ND", fl="Non Seasonal")
electricity_nd

#final results
traf_el_data <- data.frame("Traffic" = traffic_nd, "Electricity" = electricity_nd, row.names=c("SNaive", "Theta", "ARIMA"))
traf_el_data

#fred data preparation
year_data <- read.csv("~/Scripts/Machine Learning/Project/R/fred/yearly.csv")
quarter_data <- read.csv("~/Scripts/Machine Learning/Project/R/fred/quarterly.csv")
month_data <- read.csv("~/Scripts/Machine Learning/Project/R/fred/monthly.csv")
week_data <- read.csv("~/Scripts/Machine Learning/Project/R/fred/weekly.csv")
day_data <- read.csv("~/Scripts/Machine Learning/Project/R/fred/daily.csv")

year_h <- 6
quarter_h <- 8
month_h <- 18
week_h <- 13
day_h <- 14

year_m <- 1
quarter_m <- 4
month_m <- 12
week_m <- 7
day_m <- 1

fred_preprocess <- function(yy, h){
  n <- dim(yy)
  y_length <- apply(yy, 1, function(x) length(which(!is.na(x))))
  y_length <- unlist(y_length)
  y <- t(yy[,2:n[2]])
  n <- dim(y)
  y_length <- y_length - h - 1
  y_train <- matrix(0,n[1]-h,n[2])
  y_test <- matrix(0,h,n[2])
  y_max <- matrix(0,1,n[2])
  #y_min <- matrix(0,1,n[2])
  for (j in 1:n[2]){
    l <- y_length[j]
    y_max[1,j] <- max(abs(y[1:l,j]))
    y_train[1:l,j] <- y[1:l,j] / y_max[1,j]
    y_test[,j] <- y[(l+1):(l+h),j]
  }
  
  res = list("train" = y_train, "test" = y_test, "len" = y_length, "max" = y_max) 
  
  return (res)
}

theta <- function(data, leng, h, m){
  len <- unlist(leng)
  n <- length(len)
  
  y_pred <- matrix(0,h,n)
  for (i in 1:n){
    y <- data[1:len[i],i]
    if (len[i] > 5000){
      y <- data[(len[i]-4000):len[i],i]
    }
    if (m == 1){
      y <- ts(y)
      y_pred[,i] <- dotm(y, h, s=NULL)$mean
      #y_pred[,i] <- thetaf(y, h)$mean
    } else{
      n_end <- len[i] %/% m
      m_end <- len[i] %% m
      if (m_end == 0){
        m_end = m
      } else{
        n_end <- n_end + 1
      }
      y <- ts(y, start=c(1,1), end=c(n_end,m_end), frequency=m)
      
      if (len[i] < (2*m)){
      
      } else{
        y_pred[,i] <- dotm(y, h, s="additive", par_ini=c(0.5,0.5,2))$mean
      }
      #y_pred[,i] <- thetaf(y, h)$mean
    }
  }
  
  return (y_pred)
}


fred_forecast <- function(y_train, y_test, y_length, y_max, y_h, y_m, metric, fl="Seasonal"){
  
  y_SNaive_pred <- SNaive(y_train, y_length, y_h, y_m, fl=fl)
  y_theta_pred <- theta(y_train, y_length, y_h, y_m)
  y_arima_pred <- arima_forecast(y_train, y_length, y_h, y_m)
  
  n <- dim(y_SNaive_pred)
  for (j in 1:n[2]){
    y_SNaive_pred[,j] <- y_SNaive_pred[,j] * y_max[1,j]
    y_theta_pred[,j] <- y_theta_pred[,j] * y_max[1,j]
    y_arima_pred[,j] <- y_arima_pred[,j] * y_max[1,j]
  }
  
  if (metric == "sMAPE"){
    SNaive_smape <- sMAPE_average(y_test, y_SNaive_pred)
    theta_smape <- sMAPE_average(y_test, y_theta_pred)
    arima_smape <- sMAPE_average(y_test, y_arima_pred)
    
    res <- c(SNaive_smape, theta_smape, arima_smape)
  }
  
  return (res)
}

#prediction and metric evaluation
data <- fred_preprocess(year_data, year_h)
year_train <- data$train
year_test <- data$test
year_length <- data$len
year_max <- data$max

year_smape <- fred_forecast(year_train, year_test, year_length, year_max, year_h, year_m, "sMAPE", fl="Non Seasonal")
year_smape

  
data <- fred_preprocess(quarter_data, quarter_h)
quarter_train <- data$train
quarter_test <- data$test
quarter_length <- data$len
quarter_max <- data$max

quarter_smape <- fred_forecast(quarter_train, quarter_test, quarter_length, quarter_max, quarter_h, quarter_m, "sMAPE")
quarter_smape

data <- fred_preprocess(month_data, month_h)
month_train <- data$train
month_test <- data$test
month_length <- data$len
month_max <- data$max

month_smape <- fred_forecast(month_train, month_test, month_length, month_max, month_h, month_m, "sMAPE")
month_smape           
    
data <- fred_preprocess(week_data, week_h)
week_train <- data$train
week_test <- data$test
week_length <- data$len
week_max <- data$max

week_smape <- fred_forecast(week_train, week_test, week_length, week_max, week_h, week_m, "sMAPE")
week_smape  

#final results
fred_data <- data.frame("Year" = year_smape, "Quarter" = quarter_smape, "Month" = month_smape, "Week" = week_smape, row.names=c("SNaive", "Theta", "ARIMA"))
fred_data
    
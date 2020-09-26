
# setwd('/Users/cavriends/Dropbox/ESE/MSc Econometrics/Thesis/Bayesian VARs/Code/Jupyter/rcode/')

# install.packages('vars')
# install.packages('BVAR')
library(vars)
library(BVAR)

fred_ds = read.csv('../data/normal_7_fred.csv')

train = 200
complete = dim(fred_qd)[1]
complete_y = fred_ds[,c(2,3,4,5,6,7,8)]
h_steps = 8
lags = 4

transformed_y <- fred_transform(complete_y, codes=c(5,6,2,6,6,5,5))
number_of_predictions = dim(transformed_y)[1] - train

calculate_msfe <- function(h_steps, y_pred, y_true, train) {
  number_of_predictions = length(y_pred) 
  msfe_list = list()
  for (h in 1:h_steps) {
    accumulated_error = c()
    for (i in 1:(number_of_predictions-h+1)) {
      accumulated_error = rbind(accumulated_error, (y_pred[[i]][h,] - y_true[train+(i-1)+h,])^2) 
      if (i == number_of_predictions-h+1) {
        msfe_list = append(msfe_list, list(mean(colMeans(accumulated_error))))
      }
    }
  }
  return(msfe_list)
}

### VAR-OLS ###

predict_list = list()

for (i in train:(dim(transformed_y)[1]-1)) {
  
  y = transformed_y[1:i,]
  var_fit = VAR(y, lags=lags)
  predict_result = predict(var_fit, n.ahead=h_steps)$fcst
  
  predictions = list(cbind(predict_result$GDPC1))
    
  predictions = c()
  
  for (m in 1:dim(y)[2]) {
    
    predictions = cbind(predictions, predict_result[[m]][,1])
    
  }
  
  predict_list = append(predict_list, list(predictions))
  
}

msfe_var = calculate_msfe(8, predict_list, transformed_y, train)

#### BVAR with Minnesota prior ####

predict_list = list()

for (i in train:(dim(transformed_y)[1]-1)) {
  
  y = transformed_y[1:i,]
  bvar_fit = bvar(data = y,
                  lags = lags,
                  fcast= bv_fcast(h_steps),
                  priors = bv_priors(hyper = 'full',
                                     mn = bv_minnesota(lambda = bv_lambda(),
                                                       alpha = bv_alpha(),
                                                       psi = bv_psi(), # bv_psi(mode=c(rep(0.5,6))), # naive solution if
                                                                                                     # automtic selection fails
                                                       var = 10000000,
                                                       b = 1)),
                  verbose=FALSE)
  
  predict_result = predict(bvar_fit)
  
  predictions = c()
  
  for (h in 1:h_steps) {
   
    predictions = rbind(predictions, colMeans(predict_result$fcast[,h,]))
    
  }
  
  predict_list = append(predict_list, list(predictions))
  
  if (i %% 10 == 0) {
    cat("\014")
    print("Progress of BVAR with Minnesota prior...")
  }
  
  print(sprintf("Progress %i/%i", i-train+1, number_of_predictions))
  
}

msfe_bvar = calculate_msfe(8, predict_list, transformed_y, train)

pred = c()

for (i in 1:(length(predict_list)-1)) {
  
  pred = rbind(pred, (predict_list[[i]][2,] - transformed_y[train+i,])^2)
  
}

mean(colMeans(pred))



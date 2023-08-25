set.seed(123)
rm(list = ls())
data = read.csv('C:/Users/ivan/Desktop/Coding/corunaRealEstateMarket/kk.csv')
a = b = c = 1
c=-1

x1 = data$tamano
x1 = x1-min(x1) + 1
x2 = data$n_habitaciones
y = data$y

# fit model to the data
fm = nls(y ~ a + b/(x1^0.002) + c*x2, start = list(a= mean(y), b = 2, c=1))
pred = predict(fm, list(x = c(x1,x2)))
sqrt(sum((y-pred)^2)/length(y))

plot(x1,y, pch='.', cex = 2)
x_plot = seq(0,500,0.1)
predicted_y <- predict(fm, newdata = data.frame(x1 = x_plot))
lines(x_plot, predicted_y, col = "green")

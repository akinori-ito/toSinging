library(tuneR)
library(WorldR)

x<-readWave("test.wav")
xw <- world.analysis(x)
org_F0 <- xw$F0
new_F0 <- rep(mean(xw$F0),length(org_F0))
new_F0[org_F0==0] <- 0
xw$F0 <- new_F0
y <- world.synthesis(xw)
yw <- world.analysis(y)
yw$F0 <- org_F0
z <- world.synthesis(yw)


library(ggplot2)

P<-read.table("predicted-HN-values1.tsv",sep="",header=F)
N<-read.table("nondriver-values1.tsv",sep="",header=F)
D<-read.table("known-HN-values1.tsv",sep="",header=F)

# Converted into list
Ps = unlist(P)
Non = unlist(N)
Ds = unlist(D)

#Baseline
t.test(Ds,Non)
#Predicted Driver Genes VS Non Driver Genes
t.test(Ps,Non)
t.test(Non)

dat1 <- data.frame(dens1 = c(Ps), lines1 = rep(c("Predicted H&N Driver Genes"),by=length(Ps)))
dat2 <- data.frame(dens2 = c(Ds), lines2 = rep(c("H&N Driver Genes"),by=length(Ds)))
dat3 <- data.frame(dens3 = c(Non), lines3 = rep(c("Non-Driver Genes"),by=length(Non)))

dat1$veg <- 'Predicted H&N Driver Genes'
dat2$veg <- 'H&N Driver Genes'
dat3$veg <- 'Non-Driver Genes'

colnames(dat1) <- c("x","Y")
colnames(dat2) <- c("x","Y")
colnames(dat3) <- c("x","Y")

# Plot each histogram 

g1 <- ggplot(dat1, aes(dat1$x, fill = dat1$Y)) +
  geom_histogram(bins = 150,alpha = 0.3, color="blue",
                 aes(y = (..count..)/sum(..count..)), position = 'identity', binwidth=0.1) +
  scale_x_continuous(trans='log10', limits = c(1e-8, 1e-4)) +
  scale_y_continuous(labels = percent, limits = c(0,1)) +
  labs(x="", y="") +
  theme(panel.border = element_rect(colour = "black"),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black")) +
  theme_bw()+theme(legend.title=element_blank())

g2 <- ggplot(dat2, aes(dat2$x, fill = dat2$Y)) +
  geom_histogram(bins = 150,alpha = 0.3, color="red", aes(y = (..count..)/sum(..count..)),
                 position = 'identity', binwidth=0.1) +
  scale_x_continuous(trans='log10', limits = c(1e-8, 1e-4)) +
  scale_y_continuous(labels = percent, limits = c(0,1)) +
  labs(x="", y="Percentage") +
  theme(panel.border = element_rect(colour = "black"),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black")) +
  theme_bw()+theme(legend.title=element_blank())

g3 <- ggplot(dat3, aes(dat3$x, fill = dat3$Y)) +
  geom_histogram(bins = 150,alpha = 0.3, color="green",
                 aes(y = (..count..)/sum(..count..)), position = 'identity', binwidth=0.1) +
  scale_x_continuous(trans='log10', limits = c(1e-8, 1e-4)) +
  scale_y_continuous(labels = percent, limits = c(0,1)) +
  labs(x="Somatic Mutation Frequency", y="") +
  theme(panel.border = element_rect(colour = "black"),
        panel.grid.minor = element_blank(),
        axis.line = element_line(colour = "black")) +
  theme_bw()+theme(legend.title=element_blank())

#library(gridExtra)
grid.arrange(g1, g2, g3, ncol = 1)

# another trying to combine side-by-side
ggarrange(g1, g2, g3, ncol = 1)

#library(plotrix)
l <- list(dat1$x,dat2$x,dat3$x)
multhist(l)

# # To combine all histograms
data = rbind(dat1, dat2, dat3)

#ggplot(melt(data), aes(data$x, fill = data$Y)) + geom_histogram(position = "dodge")
# ggplot(data, aes(data$x, fill = data$Y)) +
#   geom_histogram(position="dodge2")
# 
# # Draw them all as a one plot
# ggplot(data, aes(data$x, fill = data$Y)) +
#   geom_histogram(bins = 150,alpha = 0.3, aes(y = 3*(..count..)/sum(..count..)),
#                  position = 'identity') +
#   scale_y_continuous(labels = percent, limits = c(0, 1)) +
#   labs(x="Somatic Mutation Frequency", y='Percentage') +
#   theme(panel.border = element_rect(colour = "black"),
#         panel.grid.minor = element_blank(),
#         axis.line = element_line(colour = "black")) +
#   theme_bw()+theme(legend.title=element_blank())+
#   facet_wrap(~Y, ncol = 1, scales = 'free') +
#   theme(panel.spacing = unit(0, 'lines'))+theme(
#   strip.background = element_blank(),
#   strip.text.x = element_blank()
# )


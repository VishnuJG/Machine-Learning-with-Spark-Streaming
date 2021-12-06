batchsize=[100,150,200,250,300]
SGD=[0.4907 0.396 0.9566 0.924 0.99]

plot(batchsize,SGD,'r*-')
xlabel("Batch Size")
ylabel("Final Accuracy")
title("Linear Model(SGD)")

figure

Burnacc=[0.54133 0.655 0.96 0.944 0.966]
plot(batchsize,Burnacc,'b*-')
xlabel("Batch Size")
ylabel("Final Accuracy")
title("Burnoulli model")

figure

kmeans=[0.5492 0.6408 0.561 0.554 0.566]
plot(batchsize,kmeans,'g*-')
xlabel("Batch Size")
ylabel("Final Accuracy")
title("K means Clustering")

figure

naive=[0.587 0.6573 0.9733 0.97 0.97]
plot(batchsize,naive,'y*-')
xlabel("Batch Size")
ylabel("Final Accuracy")
title("Naive Bayesian model")




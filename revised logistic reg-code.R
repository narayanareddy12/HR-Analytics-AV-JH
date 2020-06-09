rm(list = ls())

#1. Import the data (train data and will do test data once model built) ====
data1 = read.csv(file.choose())  # please upload 'train.csv' file

#2. Understanding & Pre processing the data ====

#2a. Dimensions, structure & summary of data
dim(data1) #dimenssions of data
str(data1) #structure of data
summary(data1) #Summary stats of each variable

#2b. Dealing with missing values

#check for number of 'missing values' in data frame
sum(data1$city=="")                      #no mssing values in city
data1$city<- as.factor(data1$city)
sum(data1$city_development_index=="")    #no mssing values in city_development_index

sum(data1$gender=="")                    #missing values in gender
data1$gender<-as.factor(data1$gender)
data1[data1$gender=='','gender']<- 'Male' #assigning missing values to "Male"

data1$relevent_experience<-as.factor(data1$relevent_experience)
sum(data1$relevent_experience=='')        #no missing values in relevant experience

data1$enrolled_university<-as.factor(data1$enrolled_university)
data1[data1$enrolled_university=='','enrolled_university']<-'no_enrollment'  #assigning missing values to 'no enrollment

data1$education_level<-as.factor(data1$education_level)
data1[data1$education_level=='','education_level']<-'Graduate'      #assinging missing value to"Graduate"

data1$major_discipline<-as.factor(data1$major_discipline)
data1[data1$major_discipline=='','major_discipline']<-'STEM'        #assigning missing valu to "STEM"

data1$experience<-as.factor(data1$experience)
data1[data1$experience=='','experience']<-'>20'                     #assigning missing values to ">20"

data1$company_size<- as.factor(data1$company_size)
#vale<- c("50-99","100-500","10000+","10/49")
v<- which(data1$company_size=='',arr.ind=TRUE)                     #to get index of missing values
for ( i in v){                                                    # applying missing values through for loop
  if (i%%4==0) {
    data1$company_size[i]<- '100-500'
  } else if (i%%3==0) {
    data1$company_size[i]<- '10000+'
  } else if (i%%2==0) {
    data1$company_size[i]<- '10/49'
  } else{
    data1$company_size[i]<- '50-99'
  }
}  

data1$company_type<- as.factor(data1$company_type)          
sum(data1$company_type=='')
data1[data1$company_type=='','company_type']<- 'Pvt Ltd'         #assigning missing values to "Pvt Ltd"

data1$last_new_job<- as.factor(data1$last_new_job)
data1[data1$last_new_job=='','last_new_job']<- '1'               #assigning missing values to '1'

#3. Split the data into train validation and test ====

library(caret)
set.seed(1234)

table(data1$target) #Distribution of levels in target data
table(data1$target) / nrow(data1)

# Finding the train,validation rows & subset from data
train_rows = createDataPartition(data1$target, p = 0.7, list = F)
train_data = data1[train_rows, ]
validation_data = data1[-train_rows, ]

# Check the dimensions of train, validation
dim(train_data) #dimensions of train data
dim(validation_data) #dimensions of test data

dim(data1) #dimensions of original data
dim(train_data)[1] + dim(validation_data)[1] #sum of rows in train, validation

#check for distribution of levels in split data
prop.table(table(data1$target))
prop.table(table(train_data$target))
prop.table(table(validation_data$target))


#4. Build logistic regression model & find the predictions ====
#log_reg = glm(target ~ ., data = train_data, family = "binomial")
#summary(log_reg)

log_reg <- glm(target ~ .-city, data=train_data, family="binomial")
log_reg$xlevels[["city"]] <- union(log_reg$xlevels[["city"]], levels(validation_data$city))

#mod2 <- glm(target~., data=train_data[,!colnames(train_data) %in% c("city")], family="binomial")

#Finding beta values
train_beta = predict(log_reg,train_data)
train_beta[1]
train_beta
1 / (1+exp(-train_beta[1]))

#Finding probabilities
train_probabilities = predict(log_reg,train_data,type = "response")
train_probabilities[1]

#Finding predictions
levels(data1$target) #understand the order of levels
train_prediction = ifelse((train_probabilities<0.5),0, 1)

#Validation predictions
val_probabilities = predict(log_reg,validation_data,type = "response")
val_prediction = ifelse((val_probabilities<0.5),0, 1)

#5. Building confusion matrix ====
con_mat = table(train_prediction,train_actual = train_data$target)
val_con_mat = table(val_prediction,val_actual = validation_data$target)

# Finding required metrics on train data
train_accuracy = sum(diag(con_mat)) / sum(con_mat)
train_recall = con_mat[2,2] / sum(con_mat[,2])
train_precision = con_mat[2,2] / sum(con_mat[2,])

# Finding metrics using inbuilt function
#library(caret) #req function available in caret package
confusionMatrix(con_mat)
confusionMatrix(con_mat,positive = '1')

# Finding validation metrics
confusionMatrix(val_con_mat,positive = '1')

#6. Identifying collinear variables using 'VIF' & important variables using 'Step-AIC' ====

# 'VIF' for collinear variables
library(car)
vif(log_reg)

# 'Step-AIC' for important variables

library(MASS)
m = stepAIC(log_reg) #using Step-AIC on logistic reg model
summary(m) #check the summary of logistic + Step-AIC model
m$call #get the syntax for best variable model

# Building logistic regression using variables from Step-AIC
log_reg_step_aic = glm(formula = target ~ enrollee_id + city_development_index + 
                         relevent_experience + enrolled_university + education_level + 
                         experience + company_size + last_new_job, family = "binomial", 
                       data = train_data)

#7. Finding train and validation pobabilities for the log+step-AIC model ====

# Finding probabilities
step_train_probailities = predict(log_reg_step_aic,train_data,type = "response")
step_val_probailities = predict(log_reg_step_aic,validation_data,type = "response")

# Finding predictions
step_train_predictions = ifelse((step_train_probailities<0.5), 0, 1)
step_val_predictions = ifelse((step_val_probailities<0.5), 0, 1)

# Finding confusion matrices
step_con_mat = table(step_train_predictions,train_actual = train_data$target)
step_val_con_mat = table(step_val_predictions,val_actual = validation_data$target)

# Finding metrics
confusionMatrix(step_con_mat,positive = '1')
confusionMatrix(step_val_con_mat,positive = '1')

#8. Using ROCR curves to find the best cut-off value ====

library(ROCR)

#Creating prediction object for ROCR
rocpreds = prediction(step_train_probailities, train_data$target)


# Extract performance measures (True Positive Rate and False Positive Rate) using the "performance()" function from the ROCR package
# The performance() function from the ROCR package helps us extract metrics such as True positive rate, False positive rate etc. from the prediction object, we created above.
# Two measures (y-axis = tpr, x-axis = fpr) are extracted
perf = performance(rocpreds, measure="tpr", x.measure="fpr")
slotNames(perf)

perf
# Plot the ROC curve using the extracted performance measures (TPR and FPR)
plot(perf, col = rainbow(10), colorize=T, print.cutoffs.at=seq(0,1,0.05))

# Extract the AUC score of the ROC curve and store it in a variable named "auc"
# Use the performance() function on the prediction object created above using the ROCR package, to extract the AUC score
perf_auc = performance(rocpreds,  measure="auc")

# Access the auc score from the performance object
auc = perf_auc@y.values[[1]]
auc

# For different threshold values identifying the tpr and fpr
cutoffs = data.frame(cut= perf@alpha.values[[1]], fpr= perf@x.values[[1]], 
                     tpr=perf@y.values[[1]])

# Sorting the data frame in the decreasing order based on tpr
cutoffs = cutoffs[order(cutoffs$tpr, cutoffs$fpr, decreasing=TRUE),]
head(cutoffs)
class(perf)

# Plotting the true positive rate and false negative rate based on the cutoff       
# increasing from 0.05-0.1
plot(perf, colorize = TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))

## Choose a Cutoff Value
# Based on the trade off between TPR and FPR depending on the business domain, a call on the cutoff has to be made.
# A cutoff of 0.05 can be chosen which is in conservative area

#9. Using best cutoff value, find new predictions & new metrics ====
# Finding predictions
new_step_train_predictions = ifelse((step_train_probailities<0.45), 0, 1)
new_step_val_predictions = ifelse((step_val_probailities<0.45), 0, 1)

# Finding confusion matrices
new_step_con_mat = table(new_step_train_predictions,train_actual = train_data$target)
new_step_val_con_mat = table(new_step_val_predictions,val_actual = validation_data$target)

# Finding metrics
confusionMatrix(new_step_con_mat,positive = "1")
confusionMatrix(new_step_val_con_mat,positive = "1")



#------------------------ final model is------------#
#log_reg_step_aic = glm(formula = target ~ enrollee_id + city_development_index + 
#                         relevent_experience + enrolled_university + education_level + 
#                         experience + company_size + last_new_job, family = "binomial", 
#                       data = train_data)



#----------------------------Working for test data---------------#


test = read.csv(file.choose())  # please upload 'test.csv' file

dim(test) #dimenssions of data
str(test) #structure of data
summary(test) #Summary stats of each variable

#check for number of 'missing values' in data frame
#process same as train data
test$city<- as.factor(test$city)         #categorical casting
sum(test$city=="")                      #no mssing values in city
test$gender<-as.factor(test$gender)     #categorical casting
sum(test$gender=="")
test[test$gender=='','gender']<- 'Male' #assigning missing values to "Male"

test$relevent_experience<-as.factor(test$relevent_experience)   #categorical casting
sum(test$relevent_experience=='')        #no missing values in relevant experience

test$enrolled_university<-as.factor(test$enrolled_university)
test[test$enrolled_university=='','enrolled_university']<-'no_enrollment'  #assigning missing values to 'no enrollment

test$education_level<-as.factor(test$education_level)
test[test$education_level=='','education_level']<-'Graduate'      #assinging missing value to"Graduate"

test$major_discipline<-as.factor(test$major_discipline)
test[test$major_discipline=='','major_discipline']<-'STEM'        #assigning missing valu to "STEM"

test$experience<-as.factor(test$experience)
test[test$experience=='','experience']<-'>20'                     #assigning missing values to ">20"

test$company_size<- as.factor(test$company_size)
#vale<- c("50-99","100-500","10000+","10/49")
vtest<- which(test$company_size=='',arr.ind=TRUE)                     #to get index of missing values
for ( i in vtest){                                                    # applying missing values through for loop
  if (i%%4==0) {
    test$company_size[i]<- '100-500'
  } else if (i%%3==0) {
    test$company_size[i]<- '10000+'
  } else if (i%%2==0) {
    test$company_size[i]<- '10/49'
  } else{
    test$company_size[i]<- '50-99'
  }
}  

test$company_type<- as.factor(test$company_type)          
sum(test$company_type=='')
test[test$company_type=='','company_type']<- 'Pvt Ltd'         #assigning missing values to "Pvt Ltd"

test$last_new_job<- as.factor(test$last_new_job)
test[test$last_new_job=='','last_new_job']<- '1'               #assigning missing values to '1'

#test data prediction
test_pred = predict(log_reg_step_aic,test,type = "response")

# Finding predictions
test_predictions = ifelse((test_pred<0.45), 0, 1)

#--------------------preparing output file------------
enrollee_id<- test$enrollee_id
output_df<- data.frame(enrollee_id)  #data frame created
output_df$target<-test_predictions
write.csv(output_df,file='revised_sample_submission.csv',row.names=FALSE)

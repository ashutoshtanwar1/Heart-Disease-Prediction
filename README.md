# Heart Disease prediction using Machine Learning.

Introduction

Heart diseases is a term covering any disorder of the heart.
Heart diseases have become a major concern to deal with as studies show that the number of deaths due to heart diseases have increased significantly over the past few decades in India, in fact it has become the leading cause of death in India.

A study shows that from 1990 to 2016 the death rate due to heart diseases have increased around 34 per cent from 155.7 to 209.1 deaths per one lakh population in India.

Thus preventing Heart diseases has become more than necessary.
Good data-driven systems for predicting heart diseases can improve the entire research and prevention process, making sure that more people can live healthy lives.
This is where Machine Learning comes into play.
Machine Learning helps in predicting the Heart diseases, and the predictions made are quite accurate.

Problem Description :

A dataset is formed by taking into consideration some of the information of 920 individuals.
The problem is : based on the given information about each individual we have to calculate that whether that individual will suffer from heart disease.

# Dataset :

The dataset consists of 920 individuals data.
There are 15 columns in the dataset, however the first column name is not a good parameter as far as machine learning is considered so, there are effectively 14 columns.

1.	Age : displays the age of the individual.
2.	Sex : displays the gender of the individual using the following   format : 1 = male
          0 = female.
3.	Chest-pain type : displays the type of chest-pain experienced by the individual using the following format :
           1 = typical angina
           2 = atypical angina
           3 = non - anginal pain
           4 = asymptotic
4.	Resting Blood Pressure : displays the resting blood pressure value of an individual in mmHg (unit)
5.	Serum Cholestrol : displays the serum cholestrol in mg/dl (unit)
6.	Fasting Blood Sugar : compares the fasting blood sugar value of an individual with 120mg/dl. 
   If fasting blood sugar > 120mg/dl then : 1  (true)
                                else : 0   (false)
7.	Resting ECG : 
              0 = normal
              1 = having ST-T wave abnormality
              2 = left ventricular hyperthrophy
8.	Max heart rate achieved : displays the max heart rate achieved by an individual.
9.	Exercise induced angina : 
              1 = yes
              0 = no
10.	ST depression induced by exercise relative to rest : displays the value which is integer or float.
11.	Peak exercise ST segment : 
              1 = upsloping
              2 = flat
              3 = downsloping
12.	Number of major vessels (0-3) colored by flourosopy : displays the value as integer or float.
13.	Thal : displays the thalassemia : 
              3 = normal
              6 = fixed defect
              7 = reversable defect
14.	Diagnosis of heart disease : Displays whether the individual is suffering from heart disease or not : 
              0 = absence
              1,2,3,4 = present.

# Why these parameters:
	Age: 
Age is the most important risk factor in developing cardiovascular or heart diseases, with approximately a tripling of risk with each decade of life.[24] Coronary fatty streaks can begin to form in adolescence.[25] It is estimated that 82 percent of people who die of coronary heart disease are 65 and older.[26] Simultaneously, the risk of stroke doubles every decade after age 55.

	Sex: 
Men are at greater risk of heart disease than pre-menopausal women.[24][30] Once past menopause, it has been argued that a woman's risk is similar to a man's[30] although more recent data from the WHO and UN disputes this.[24] If a female has diabetes, she is more likely to develop heart disease than a male with diabetes.

	Angina (Chest Pain):
Angina is chest pain or discomfort caused when your heart muscle doesn't get enough oxygen-rich blood. It may feel like pressure or squeezing in your chest. The discomfort also can occur in your shoulders, arms, neck, jaw, or back. Angina pain may even feel like indigestion.

	Resting Blood Pressure:
Over time, high blood pressure can damage arteries that feed your heart. High blood pressure that occurs with other conditions, such as obesity, high cholesterol or diabetes, increases your risk even more.

	Serum Cholestrol: 
A high level of low-density lipoprotein (LDL) cholesterol (the "bad" cholesterol) is most likely to narrow arteries. A high level of triglycerides, a type of blood fat related to your diet, also ups your risk of heart attack. However, a high level of high-density lipoprotein (HDL) cholesterol (the "good" cholesterol) lowers your risk of heart attack.

	Fasting Blood Sugar:
Not producing enough of a hormone secreted by your pancreas (insulin) or not responding to insulin properly causes your body's blood sugar levels to rise, increasing your risk of heart attack.

	Resting ECG:
For people at low risk of cardiovascular disease, the USPSTF concludes with moderate certainty that the potential harms of screening with resting or exercise ECG equal or exceed the potential benefits. For people at intermediate to high risk, current evidence is insufficient to assess the balance of benefits and harms of screening.

	Max heart rate achieved:
The increase in the cardiovascular risk, associated with the acceleration of heart rate, was comparable to the increase in risk observed with high blood pressure. It has been shown that an increase in heart rate by 10 beats per minute was associated with an increase in the risk of cardiac death by at least 20%, and this increase in the risk is similar to the one observed with an increase in systolic blood pressure by 10 mm Hg.

	Exercise induced angina:
The pain or discomfort associated with angina usually feels tight, gripping or squeezing, and can vary from mild to severe. Angina is usually felt in the centre of your chest, but may spread to either or both of your shoulders, or your back, neck, jaw or arm. It can even be felt in your hands.
o			Types of Angina
a.	Stable Angina / Angina Pectoris
b.	Unstable Angina
c.	Variant (Prinzmetal) Angina
d.	Microvascular Angina

	ST depression induced by exercise relative to rest :

	Peak exercise ST segment:
A treadmill ECG stress test is considered abnormal when there is a horizontal or down-sloping ST-segment depression ≥ 1 mm at 60–80 ms after the J point. Exercise ECGs with up-sloping ST-segment depressions are typically reported as an ‘equivocal’ test. In general, the occurrence of horizontal or down-sloping ST-segment depression at a lower workload (calculated in METs) or heart rate indicates a worse prognosis and higher likelihood of multi-vessel disease. The duration of ST-segment depression is also important, as prolonged recovery after peak stress is consistent with a positive treadmill ECG stress test. Another finding that is highly indicative of significant CAD is the occurrence of ST-segment elevation > 1 mm (often suggesting transmural ischaemia); these patients are frequently referred urgently for coronary angiography.


# Model Training and Prediction : 
We can train our prediction model by analyzing existing data because we already know whether each patient has heart disease. This process is also known as supervision and learning. The trained model is then used to predict if users suffer from heart disease. The training and prediction process is described as follows:

# Splitting: 
First, data is divided into two parts using component splitting. In this experiment, data is split based on a ratio of 80:20 for the training set and the prediction set. The training set data is used in the logistic regression component for model training, while the prediction set data is used in the prediction component.

# Logistic Regression:
Logistic regression is used to describe data and to explain the relationship between one dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables.

# Prediction:
The two inputs of the prediction component are the model and the prediction set. The prediction result shows the predicted data, actual data, and the probability of different results in each group.

# Evaluation: 
The confusion matrix, also known as the error matrix, is used to evaluate the accuracy of the model.

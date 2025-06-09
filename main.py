import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pickle




data = pd.read_csv('heart.csv')




#cleaning the data
data.drop_duplicates(inplace=True)
data.duplicated().sum()
print("Number of duplicate rows:", data.duplicated().sum())  # this prints 0 as there are no duplicates




data.dropna(inplace=True)
#print("Number of missing values in each column:\n", data.isnull().sum()) # this prints 0 for all the columns as there are no missing values






print(data[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'MaxHR', 'ExerciseAngina', 'Oldpeak']].head(6)) # used this comand to see how the columns are and studdy the overview fo the DS


#----------------------------------------------------------- data  visualization------------------------------------------------------------
heart_counts = data['HeartDisease'].value_counts()
plt.bar(heart_counts.index, heart_counts.values, color=['green', 'red'])
plt.title("Heart Disease Distribution")
plt.xlabel("0 = No Disease, 1 = Disease")
plt.ylabel("Number of Patients")
plt.show()


# Filter rows based on target
has_disease = data[data['HeartDisease'] == 1]
no_disease = data[data['HeartDisease'] == 0]


# Create the plot
plt.figure(figsize=(10,6))
plt.scatter(has_disease['Age'], has_disease['Cholesterol'], color='red', label='Heart Disease', alpha=0.6)
plt.scatter(no_disease['Age'], no_disease['Cholesterol'], color='green', label='No Heart Disease', alpha=0.6)
                                                                              # after studying this grapgh we found that there are
                                                                              #  cases in which people have high cholestrol but still
                                                                              # NO SIGN OF HEART FAILURE but there are also some cases in which people with
                                                                              #  less to no cholestrol having a heart failure
                                                                              #  HENCE WE KNOW THAT CHOLESTROL ISNT JUST A FACTOR AFFECTING THE HEART FAILURE


plt.title("Age vs Cholesterol - Colored by Heart Disease")
plt.xlabel("Age")
plt.ylabel("Cholesterol")
plt.legend()
plt.grid(True)
plt.show()










# Calculate average Oldpeak for each group
avg_oldpeak_has_disease = has_disease['Oldpeak'].mean()
avg_oldpeak_no_disease = no_disease['Oldpeak'].mean()


# Bar chart setup
plt.figure(figsize=(10,6))
plt.bar(['No Heart Disease', 'Heart Disease'],                        
        [avg_oldpeak_no_disease, avg_oldpeak_has_disease],
        color=['green', 'red'])


                                                        # in this grapgh we saw that people with heart disease have a higher oldpeak value and its really prominent
                                                        #hence making old peak one of the key variables in a heart disease


plt.title("Average Oldpeak - Bar Graph by Heart Disease")
plt.ylabel("Average Oldpeak")
plt.grid(axis='y')
plt.show()






has_disease = data[data['HeartDisease'] == 1]
no_disease = data[data['HeartDisease'] == 0]


# Create the plot
plt.figure(figsize=(10,6))
plt.scatter(has_disease['Age'], has_disease['MaxHR'], color='red', label='Heart Disease', alpha=0.6)
plt.scatter(no_disease['Age'], no_disease['MaxHR'], color='green', label='No Heart Disease', alpha=0.6)     # Max heart rate (MaxHR) is achieved during exercise and reflects how well the heart responds
                                                                                                            # to physical stress. A lower MaxHR during exertion indicates poor heart performance and can signal heart disease.
                                                                                                            # While MaxHR naturally decreases with age, an unusually low MaxHR in a middle-aged or older person is a strong early
                                                                                                            # indicator of potential heart failure. This is clearly shown in the graph, where most heart disease cases have lower MaxHR values.
                                                                             


plt.title("Age vs MaxHR - Colored by Heart Disease")
plt.xlabel("Age")
plt.ylabel('MaxHR')
plt.legend()
plt.grid(True)
plt.show()






# make something similar for chest pain type




grouped = data.groupby(['ChestPainType', 'HeartDisease']).size().unstack(fill_value=0)


# Step 2: Plot as grouped bar chart
labels = grouped.index
x = range(len(labels))
width = 0.4


plt.figure(figsize=(10,6))
plt.bar(x, grouped[0], width=width, label='No Heart Disease', color='green')
plt.bar([i + width for i in x], grouped[1], width=width, label='Heart Disease', color='red')          #ASY (asymptomatic) chest pain type is the biggest silent warning sign of heart disease.
                                                                                                      #ATA and NAP seem more common in healthy patients.






# Step 3: Add labels and styling
plt.xticks([i + width/2 for i in x], labels)
plt.xlabel("Chest Pain Type")
plt.ylabel("Number of Patients")
plt.title("Heart Disease by Chest Pain Type")
plt.legend()
plt.grid(axis='y')
plt.show()


#==============================================================xxxxxx==========================================================================================================


data_encoded = pd.get_dummies(data, columns=['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'], drop_first=True)

# =====================================
# STEP 3: Split Data into Features & Target
# =====================================
X = data_encoded.drop('HeartDisease', axis=1)
y = data_encoded['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



with open("heart_model.pkl", "wb") as file:
    pickle.dump(model, file)
# =====================================
# STEP 6: Predict on a Sample Patient
# =====================================
# Example: create a DataFrame for a single custom patient
# Create an empty DataFrame with the same columns as X_train
# Print to confirm expected columns



# print(" Model expects columns:", len(X_train.columns))
# print(" Column names:\n", list(X_train.columns))

# # Create patient input matching the model's expected features exactly
# patient_data = pd.DataFrame(columns=X_train.columns)

# patient_data.loc[0] = [
#     55,     # Age
#     140,    # RestingBP
#     260,    # Cholesterol
#     1,      # FastingBS
#     130,    # MaxHR
#     1.2,    # Oldpeak
    
#     # One-hot encoded categorical features (13 binary values)
#     1,  # Sex_M

#     1,  # ChestPainType_ATA
#     0,  # ChestPainType_NAP
#     0,  # ChestPainType_TA

#     1,  # RestingECG_Normal
#     0,  # RestingECG_ST

#     0,  # ExerciseAngina_Y

#     1,  # ST_Slope_Flat
#     0   # ST_Slope_Up
# ]

# # Make prediction
# prediction = model.predict(patient_data)
# prob = model.predict_proba(patient_data)

# print("\n Custom Patient Prediction:", " Heart Disease" if prediction[0]==1 else " No Heart Disease")
# print(" Risk Probability:", round(prob[0][1]*100, 2), "%")



# ---------------- Patient Input ----------------
print("\n🩺 Enter your health details below:")

age = int(input("Age: "))
resting_bp = int(input("Resting Blood Pressure (mm Hg): "))
cholesterol = int(input("Cholesterol Level (mg/dl): "))
fasting_bs = int(input("Fasting Blood Sugar (1 = Yes, 0 = No): "))
max_hr = int(input("Maximum Heart Rate Achieved: "))
oldpeak = float(input("Oldpeak (ST depression by exercise): "))

# ---- Sex Encoding ----
sex_input = input("Sex (M/F): ").strip().upper()
if sex_input == "M":
    sex_m = 1
else:
    sex_m = 0

# ---- Chest Pain Type Encoding ----
print("\n--- Chest Pain Type ---")
cp_input = input("Type (ASY / ATA / NAP / TA): ").strip().upper()
if cp_input == "ATA":
    cp_ata = 1
    cp_nap = 0
    cp_ta = 0
elif cp_input == "NAP":
    cp_ata = 0
    cp_nap = 1
    cp_ta = 0
elif cp_input == "TA":
    cp_ata = 0
    cp_nap = 0
    cp_ta = 1
else:  # ASY or unknown
    cp_ata = 0
    cp_nap = 0
    cp_ta = 0

# ---- Resting ECG Encoding ----
print("\n--- Resting ECG ---")
ecg_input = input("Type (Normal / ST / LVH): ").strip().upper()
if ecg_input == "NORMAL":
    ecg_normal = 1
    ecg_st = 0
elif ecg_input == "ST":
    ecg_normal = 0
    ecg_st = 1
else:  
    ecg_normal = 0
    ecg_st = 0

# ---- Exercise-Induced Angina ----
angina_input = input("Exercise-Induced Angina (Y/N): ").strip().upper()
if angina_input == "Y":
    angina = 1
else:
    angina = 0

# ---- ST Slope Encoding ----
print("\n--- ST Slope ---")
slope_input = input("Type (Flat / Up / Down): ").strip().upper()
if slope_input == "FLAT":
    slope_flat = 1
    slope_up = 0
elif slope_input == "UP":
    slope_flat = 0
    slope_up = 1
else:  
    slope_flat = 0
    slope_up = 0

# ---------------- Prediction ----------------
# Assuming your model is trained and X_train is available

patient_data = pd.DataFrame(columns=X_train.columns)

# Order matters here - match it with X_train.columns
patient_data.loc[0] = [
    age,
    resting_bp,
    cholesterol,
    fasting_bs,
    max_hr,
    oldpeak,
    sex_m,
    cp_ata, cp_nap, cp_ta,
    ecg_normal, ecg_st,
    angina,
    slope_flat, slope_up
]

# Make prediction
prediction = model.predict(patient_data)
prob = model.predict_proba(patient_data)

# ---------------- Output ----------------
print("\n Prediction:", " Heart Disease" if prediction[0] == 1 else " No Heart Disease")
print(" Risk Probability:", round(prob[0][1]*100, 2), "%")

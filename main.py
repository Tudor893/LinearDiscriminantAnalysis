import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df_nsal = pd.read_csv("data_in/E_NSAL_2008-2021.csv", index_col=0)
df_popLoc = pd.read_csv("data_in/PopulatieLocalitati.csv", index_col=0)
indici = list(df_nsal)[:]

#Cerinta1
an_max = df_nsal.idxmax(axis=1)
df_cerinta1 = pd.DataFrame(
    data=an_max,
    columns=["Anul"],
    index=df_nsal.index
)
df_cerinta1.to_csv("data_out/Cerinta1.csv")

#Cerinta2
def fc(t:pd.Series):
    pop_sum = t["Populatie"].sum()
    single_year_rate = t[indici].sum() / pop_sum
    mean_rate = single_year_rate.mean()
    return pd.Series(list(single_year_rate) + [mean_rate], index=indici + ["RataMedie"])


df_merged = df_nsal.merge(df_popLoc[["Judet","Populatie"]], left_index=True, right_index=True)
df_grouped = df_merged.groupby(by="Judet").apply(func=fc, include_groups=False)
df_grouped.round(3).sort_values(by="RataMedie", ascending=False).to_csv("data_out/Cerinta2.csv")


#B
df_pacienti = pd.read_csv("data_in/Pacienti.csv", index_col=0)

X = df_pacienti.drop(columns="DECISION")
Y = df_pacienti["DECISION"]
classes = len(set(Y))

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.4, random_state=0)

#1
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
scores = lda.transform(X_train)
df_scores = pd.DataFrame(
    data=scores,
    columns=[f"SC{i+1}" for i in range(classes - 1)]
)
df_scores.index.name = "ID"
df_scores.to_csv("data_out/z.csv")

#2
seaborn.kdeplot(df_scores["SC1"], label="SC1", fill=True)
seaborn.kdeplot(df_scores["SC2"], label="SC2", fill=True)
plt.show()


#3
prediction_test = lda.predict(X_test)
conf_matrix = confusion_matrix(y_test, prediction_test)
df_conf_matrix = pd.DataFrame(conf_matrix, index=lda.classes_, columns=lda.classes_)
df_conf_matrix.to_csv("data_out/matc.csv")

acc_score = accuracy_score(y_test, prediction_test)
print(acc_score) # 0.6

X_apply = pd.read_csv("data_in/Pacienti_apply.csv", index_col=0)
prediction_apply = lda.predict(X_apply)

# Salvam predictiile pentru pacientii noi
df_prediction = pd.DataFrame({"DECISION": prediction_apply})
df_prediction.index.name = "ID"
df_prediction.to_csv("data_out/Predictii_Apply.csv")

#Extra, comparare model liniar cu model bayesian
bayes_model = GaussianNB()
bayes_model.fit(X_train, y_train)
predict_bayes = bayes_model.predict(X_test)

prediction_matrix = pd.DataFrame({
    "LDA": prediction_test,
    "Naive Bayes": predict_bayes
})
prediction_matrix.index.name = "ID"
prediction_matrix.to_csv("data_out/Models_prediction.csv")

#Aplicam si modelul Bayes pe X_apply
predict_bayes_apply = bayes_model.predict(X_apply)

df_apply_comparison = pd.DataFrame({
    "LDA": prediction_apply,
    "Naive Bayes": predict_bayes_apply
})
df_apply_comparison.index.name = "ID"
df_apply_comparison.to_csv("data_out/Apply_Models_Comparison.csv")
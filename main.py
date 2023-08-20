from sklearn import svm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tkinter import *
import os
import sys
from sklearn import metrics

plt.style.use('fivethirtyeight')
import warnings

warnings.filterwarnings('ignore')
# Rui Huang Student ID: 007717147

# set the directory for milk data
file = 'milknew.csv'
if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
elif __file__:
    application_path = os.path.dirname(__file__)

file_path = os.path.join(application_path, file)


# load file
def load_file(name):
    dt = pd.read_csv(name)
    return dt


milkdata = load_file(file_path)

empty_data = milkdata.isnull().sum()

milkdata['Quality'].replace(['low', 'medium', 'high'], [0, 1, 2], inplace=True)


# ph_quality histogram
def pH_quality_hist():
    milkdata[milkdata['Quality'] == 0].pH.plot.hist(width=0.1, bins=20, edgecolor='black', color='r', label='low')
    milkdata[milkdata['Quality'] == 1].pH.plot.hist(width=0.1, bins=20, edgecolor='black', color='y',
                                                    label='medium')
    milkdata[milkdata['Quality'] == 2].pH.plot.hist(width=0.1, bins=20, edgecolor='black', color='g', label='high')
    plt.ylabel('frequency')
    plt.xlabel('pH value')
    plt.title("pH vs frequency of different milk qualities")
    plt.legend()
    plt.show()


# Temperature_quality Scatter
def Temperature_quality_scatter():
    plt.scatter(milkdata['Temperature'], milkdata['Quality'], c=np.random.randint(0, 1000, 1059),
                s=70 * np.random.randn(1059) ** 2)
    plt.ylabel('Quality')
    plt.xlabel('Temperature')
    plt.title("Temperature vs Quality Scatter plot")
    plt.show()


# correlation heatmap

def correlation():
    sns.heatmap(milkdata.corr(), annot=True, cmap='RdYlGn', linewidths=0.2, annot_kws={'size': 20})
    fig = plt.gcf()
    fig.set_size_inches(18, 15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title("Correlation of all aspects of the milk")
    plt.show()


# training data with algorithm

train, test = train_test_split(milkdata, test_size=0.3, random_state=0, stratify=milkdata['Quality'])
train_X = train[train.columns[:7]]
train_y = train[train.columns[7:]]
test_X = test[test.columns[:7]]
test_y = test[test.columns[7:]]
X = milkdata[milkdata.columns[:7]]
y = milkdata['Quality']
model = svm.SVC(kernel='rbf', C=1, gamma=0.1)
model.fit(train_X, train_y)
prediction_1 = model.predict(test_X)

# login page

def login(info, message):
    if info[0] == "admin" and info[1] == "admin":
        message.configure(text="Log in successful")
        message.update()
        homepage()
        return True
    else:
        message.configure(text="wrong info")
        message.update()
        return False


def get_info():
    login_info = [username_tf.get(), password_tf.get()]
    return login_info

# home page


def homepage():
    login_window.destroy()
    home_window = Tk()
    home_window.title("home page")
    home_window.geometry("300x200+10+20")
    hist_btn = Button(home_window, text="pH vs Quality histogram", command=pH_quality_hist)
    hist_btn.grid(row=0, column=0)
    empty_data_lb = Label(home_window,text="Null data field:\n" + str(empty_data))
    empty_data_lb.grid(row=0,column=1,rowspan=4)
    scatter_btn = Button(home_window, text="Temperature vs Quality scatter plot", command=Temperature_quality_scatter)
    scatter_btn.grid(row=1, column=0)
    correlation_btn = Button(home_window, text="Correlation of all aspects", command=correlation)
    correlation_btn.grid(row=2, column=0)
    ml_btn = Button(home_window, text="SVM Learning model", command=ml_page)
    ml_btn.grid(row=3, column=0)
    home_window.mainloop()

# machine learning algorithm page

def ml_page():
    ml_window = Tk()
    ml_window.title("SVM Learning model")
    ml_window.geometry("300x300+10+20")
    accuracy_lb = Label(ml_window, text='')
    accuracy_lb.grid(row=8, column=0,columnspan=2)
    accuracy_lb.configure(text="Accuracy for rbf SVM is " + str(metrics.accuracy_score(prediction_1, test_y)))
    accuracy_lb.update()
    pH_lb = Label(ml_window, text="pH(1-10):")
    pH_lb.grid(row=1, column=0)
    pH_tf = Entry(ml_window, )
    pH_tf.grid(row=1, column=1)

    temp_lb = Label(ml_window, text="Temperature(0-100):")
    temp_lb.grid(row=2, column=0)
    temp_tf = Entry(ml_window)
    temp_tf.grid(row=2, column=1)

    taste_lb = Label(ml_window, text="Taste(1/0):")
    taste_lb.grid(row=3, column=0)
    taste_tf = Entry(ml_window)
    taste_tf.grid(row=3, column=1)

    odor_lb = Label(ml_window, text="Odor(1/0):")
    odor_lb.grid(row=4, column=0)
    odor_tf = Entry(ml_window)
    odor_tf.grid(row=4, column=1)

    fat_lb = Label(ml_window, text="Fat(1/0):")
    fat_lb.grid(row=5, column=0)
    fat_tf = Entry(ml_window)
    fat_tf.grid(row=5, column=1)

    turbidity_lb = Label(ml_window, text="Turbidity(1/0):")
    turbidity_lb.grid(row=6, column=0)
    turbidity_tf = Entry(ml_window)
    turbidity_tf.grid(row=6, column=1)

    color_lb = Label(ml_window, text="Colour(240-260):")
    color_lb.grid(row=7, column=0)
    color_tf = Entry(ml_window)
    color_tf.grid(row=7, column=1)
    result_lb = Label(ml_window, text='')
    result_lb.grid(row=10, column=0)
    predict_btn = Button(ml_window, text="predict", command=lambda: predict(model=model, pH_tf=pH_tf, temp_tf=temp_tf,
                                                                            taste_tf=taste_tf, odor_tf=odor_tf,
                                                                            fat_tf=fat_tf, turbidity_tf=turbidity_tf,
                                                                            color_tf=color_tf, label=result_lb))
    predict_btn.grid(row=9, column=0)
    ml_window.mainloop()

# Predict the result

def predict(model, pH_tf, temp_tf, taste_tf, odor_tf, fat_tf, turbidity_tf, color_tf, label):
    if( not pH_tf.get() or not temp_tf.get() or not taste_tf.get() or not odor_tf.get() or not fat_tf.get() or not
    turbidity_tf.get() or not color_tf.get()):
        label.configure(text="Entry missing!")
    else:
        data = [[float(pH_tf.get()), float(temp_tf.get()), int(taste_tf.get()), int(odor_tf.get()), int(fat_tf.get()),
                int(turbidity_tf.get()), int(color_tf.get())]]
        result = model.predict(data)
        if result[0] == 0:
            label.configure(text="The milk quality predicted is Low")
        if result[0] == 1:
            label.configure(text="The milk quality predicted is Medium")
        if result[0] == 2:
            label.configure(text="The milk quality predicted is High")

# login page initialize

login_window = Tk()
login_window.title('Milk quality prediction data product')
login_window.geometry("400x200+10+20")
username_lb = Label(login_window, text="Username: ")
username_tf = Entry(login_window)
password_tf = Entry(login_window, show='*')
password_lb = Label(login_window, text="password: ")
login_message = Label(login_window, text="")
username_lb.grid(row=0, column=0)
username_tf.grid(row=0, column=1)
password_lb.grid(row=1, column=0)
password_tf.grid(row=1, column=1)
login_message.grid(row=2, column=1)
login_btn = Button(login_window, text="Log in", command=lambda: login(get_info(), message=login_message))
login_btn.grid(row=3, column=1)

login_window.mainloop()

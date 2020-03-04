import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns

def plot_figure(path="/over_confident100-10.csv",x_name="Number of Classes",y_name="",mark=True):
    test = pd.read_csv(path)
    print(test[0:10])
    sns.set(style="darkgrid")
    # plt.figure()
    # Plot the responses for different events and regions
    if mark:
        a = sns.lineplot(x=x_name, y=y_name,
                         hue="model_name", style="model_type", markers=True,
                         data=test)
    else:
        a = sns.lineplot(x=x_name, y=y_name,
                         hue="model_name",style="model_name", markers=True,
                         data=test)
    print(a)
    plt.show()


def save_acc_csv(save_file, class_num, acc, model_name):
    with open(save_file, "a+", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([class_num, acc, model_name])


if __name__ == '__main__':
    # plot_figure(path="/decoder_acc100-10decay.csv",y_name="Accuracy",mark=False)
    # plot_figure(path="/decoder_acc100-10k.csv",x_name="Number of K", y_name="Accuracy", mark=False)
    # plot_figure(path="/over_confident100-20.csv",y_name="Number of Over-confidence")

    plot_figure(path="/decoder_acc100-20.csv", y_name="Accuracy", mark=True)
import pandas as pd
import io
import matplotlib.pyplot as plt
import tensorflow as tf

def formatKneeDf(path):
    df = pd.read_csv(path)

    # Adding ml_class filed that standardizes the classes to 4 (ankle, knee, elbow, other)
    def assignMlClass(x):
        accepted_classes = ["SQR","Femoral","Medial","Lateral"]
        if x in accepted_classes:
            return x
        else:
            return "other"
    df["ml_class"] = df["Scan"].apply(lambda x: assignMlClass(x))

    # Creating the patient filed that is the encrypted patient name
    df["patient"] = df["Filename"].apply(lambda x: x.split("_")[0])

    # Creating a df without other class
    dfNoOther = df[df["ml_class"]!="other"][["Filename","patient","ml_class"]]
    return dfNoOther

def formatDf(path):

    df = pd.read_csv(path)

    # Adding ml_class filed that standardizes the classes to 4 (ankle, knee, elbow, other)
    def assignMlClass(x):
        if(x =="elbow"):
            return x
        elif((x =="tibial t.") | (x =="ankle")):
            return "ankle"
        elif(x == "knee"):
            return x
        else:
            return "other"
    df["ml_class"] = df["Joint"].apply(lambda x: assignMlClass(x))

    # Creating the patient filed that is the encrypted patient name
    df["patient"] = df["Filename"].apply(lambda x: x.split("_")[0])

    # Creating a df without unknown joints
    jointsNoUnk = df[df["Joint"]!="unknown"][["Filename","patient","ml_class"]]
    return jointsNoUnk

def plot_to_image(figure):    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)

    digit = tf.image.decode_png(buf.getvalue(), channels=4)
    digit = tf.expand_dims(digit, 0)

    return digit


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(y_true, y_pred, name="Model"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n===== {name} =====")
    print("Accuracy:", round(acc, 4))
    print("Precision:", round(prec, 4))
    print("Recall:", round(rec, 4))
    print("F1 Score:", round(f1, 4))

    return acc, prec, rec, f1


def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    # 🔥 Save with dynamic filename
    filename = title.lower().replace(" ", "_") + ".png"
    plt.savefig(filename, bbox_inches='tight')

    print(f"Saved: {filename}")

    plt.show()


def compare_models(metrics):
    names = list(metrics.keys())
    accuracy = [metrics[m][0] for m in names]

    plt.figure(figsize=(6, 5))
    bars = plt.bar(names, accuracy)

    plt.title("Model Accuracy Comparison")
    plt.xlabel("Models")
    plt.ylabel("Accuracy")

    # 🔥 Add values on bars (very professional)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 3),
                 ha='center', va='bottom')

    # 🔥 Save image
    plt.savefig("model_comparison.png", bbox_inches='tight')

    print("Saved: model_comparison.png")

    plt.show()
import pickle
import matplotlib.pyplot as plt

with open("out/history_first_run.pkl", "rb") as f:
    history = pickle.load(f)

plt.figure(figsize=(8, 4))
plt.plot(history["gen_train_loss"], label="gen_train_loss")
plt.plot(history["dis_train_loss"], label="dis_train_loss")
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()

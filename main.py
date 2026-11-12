from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()

X = iris.data 
y = iris.target 


print("Features shape:", X.shape)
print("Labels shape:", y.shape)

print(iris.target_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# randomly assigns rows to the training and testing sets. Without a fixed seed, you may get a differnet
# split each time to run the script . 

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(X_train, y_train) # train the classifier using the training data
print("Classifier trained sucessfully") 

predictions = clf.predict(X_test)

print("Predictions on test data:", predictions)
print("Actual labels of test data:", y_test)

import matplotlib.pyplot as plt 
import numpy as np 

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

scatter1 = axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', alpha=0.7, s=50)
axes[0].set_xlabel('Sepal Length (cm)')
axes[0].set_ylabel('Sepal Width (cm)')
axes[0].set_title('Training Data Distribution')
axes[0].legend(*scatter1.legend_elements(), title='Classes')

test_predictions = clf.predict(X_test)
correct_predictions = (test_predictions == y_test)

correct_mask = correct_predictions 
scatter2_correct = axes[1].scatter(X_test[correct_mask, 0], X_test[correct_mask, 1], 
c=y_test[correct_mask], cmap='viridis', alpha=0.7, s=50, marker='o')


incorrect_mask = ~correct_predictions 
if np.any(incorrect_mask):
    scatter_incorrect = axes[1].scatter(X_test[incorrect_mask, 0], X_test[incorrect_mask, 1],
            c=test_predictions[incorrect_mask], cmap='viridas',
            alpha=0.7, s=80, marker='x', edgecolors='red', linewidths=2)

axes[1].set_xlabel('Sepal Length (cm)')
axes[1].set_ylabel('Sepal Width (cm)')
axes[1].set_title('Test Data: Predictions vs Actual\n(correct=@, incorrect=X)')

legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Correct'),
                   plt.Line2D([0], [0], marker='x', color='red', markersize=8, label='Incorrect')]

axes[1].legend(handles=legend_elements)

plt.tight_layout()
plt.savefig('knn_classification_results.png', dpi=150, bbox_inches='tight')
print("Visualization saved to knn_classification_results.png")

accuracy =  np.mean(correct_predictions)*100
print(f"Model Accuracy: {accuracy: .1f}%")
                




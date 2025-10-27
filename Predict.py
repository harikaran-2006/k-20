print("\n--- Prediction Tests ---")

    # 1. Known Benign
benign_sample = X[y_class == "Benign"].iloc[0:1]
print("Benign Sample:", model.predict(benign_sample))

          # 2. Known Malware
malware_sample = X[y_class == "Malware"].iloc[0:1]
print("Malware Sample:", model.predict(malware_sample))

          # 3. Mixed samples
mixed_samples = X.sample(5, random_state=42)
print("Mixed Samples:", model.predict(mixed_samples))

          # 4. All zeros synthetic
zeros_sample = pd.DataFrame(np.zeros((1, X.shape[1])), columns=X.columns)
print("Zeros Sample:", model.predict(zeros_sample))

          # 5. Half zeros, half ones
n_features = X.shape[1]
pattern_sample = pd.DataFrame(
np.concatenate([
np.zeros(n_features // 2),
np.ones(n_features - n_features // 2)
]).reshape(1, -1),
columns=X.columns
)
print("Pattern Sample:", model.predict(pattern_sample))

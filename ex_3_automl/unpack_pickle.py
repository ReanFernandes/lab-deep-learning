import pickle

# unpack the test error for the incumbesnt trajectory

with open("incumbent_trajectory_test_error_rs.pkl", "rb") as f:
    test_error = pickle.load(f)

save_path = "incumbent_trajectory_test_error_rs.txt"
with open(save_path, "w") as f:
    for i in range(len(test_error)):
        f.write(str(test_error[i]) + "\n")


import pickle

# unpack the test error for the incumbesnt trajectory

<<<<<<< HEAD
with open("all_configs_err_rs.pkl", "rb") as f:
    test_error = pickle.load(f)

save_path = "all_configs_error_rs.txt"
=======
with open("incumbent_trajectory_test_error_rs.pkl", "rb") as f:
    test_error = pickle.load(f)

save_path = "incumbent_trajectory_test_error_rs.txt"
>>>>>>> c6bc5e308f948925ff1c8434c0abdd2b199068b5
with open(save_path, "w") as f:
    for i in range(len(test_error)):
        f.write(str(test_error[i]) + "\n")


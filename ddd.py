convergence = False
while not convergence:
    label_arr = {}
    for i in range(X.shape[0]):
        distances = compute_euclid_distance(X[i, :], mu_points)
        minMuIndex = np.argmin(distances)
        if minMuIndex not in label_arr:
            label_arr[minMuIndex] = []
        label_arr[minMuIndex].append(i)

    for i in label_arr.keys():
        mu_points[i] = np.array(
            [np.sum(X[m][0] for m in label_arr[i]), np.sum(X[m][1] for m in label_arr[i])]) / np.float(
            len(label_arr[i]))

    comp_arr = np.isclose(previous_mu_points, mu_points)
    if comp_arr.all():
        convergence = True
    else:
        previous_mu_points = mu_points.copy()
    # print(f'previous: {previous_mu_points}, current: {mu_points}')
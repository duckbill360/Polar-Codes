# L
# L[:N // 2, i] = f(L[: N: 2, i + 1], L[1: N: 2, i + 1] + R[N // 2:, i])
# L[N // 2:, i] = f(R[: N // 2, i], L[: N: 2, i + 1]) + L[1: N: 2, i + 1]

# R
# R[: N: 2, i + 1] = f(R[: N // 2, i], R[N // 2:, i] + L[1: N: 2, i + 1])
# R[1: N: 2, i + 1] = f(L[: N: 2, i + 1], R[: N // 2, i]) + R[N // 2:, i]

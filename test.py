def fibonacci(n,F):
    if n in F.keys():
        return F[n]
    else:
        print(F)
        a = fibonacci(n-1, F)
        b = fibonacci(n-2, F)
        F[n] = a + b
f0 = 0
f1 = 1
F = {0:f0, 1:f1}
print(fibonacci(5, F))
print(F)
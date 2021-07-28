def func(a, **test):
    print(a)
    if test:
        print(test.keys())
        print(test.values())


a = 0
b = {'1': 1, '2': 2}
func(a, **b)

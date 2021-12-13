def test(a, b, c=3):
    print(a + b + c)

d = {'a': 1, 'b': 2}
test(**d)

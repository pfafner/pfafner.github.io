
alpha = lambda x: x+1

beta = lambda x: 2*x

cero = lambda f: lambda x: x

uno = lambda f: lambda x: f(x)

dos = lambda f: lambda x: f(f(x))

tres = lambda f: lambda x: f(f(f(x)))

sucesor = lambda n: lambda f: lambda x: n(f)(f(x))

suma = lambda a: lambda b: lambda f:lambda x: a(f)(b(f)(x))

multiplicacion = lambda a: lambda b: lambda f: lambda x: a(b(f))(x)

exponente = lambda a: lambda b: lambda f: lambda x: b(a)(f)(x)

print(exponente(uno)(tres)(alpha)(0))
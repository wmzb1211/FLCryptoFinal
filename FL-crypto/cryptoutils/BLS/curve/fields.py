"""
    Arithmetic on finite fields
"""


from abc import ABC, abstractmethod
from typing import Union, TypeVar, Sequence, Mapping
from typing_extensions import Protocol, runtime_checkable
from functools import total_ordering
import random

T = TypeVar("T")
T_Polynomial = TypeVar("T_Polynomial", bound="Polynomial")
T_FQ = TypeVar("T_FQ", bound="FQ")


"""
    Interface which all "inner values" of
    finite field elements (ints, polynomials)
    must implement
"""
@runtime_checkable
class FQVal(Protocol[T]):

    def __add__(self: T, other: T) -> T: ...
    def __radd__(self: T, other: T) -> T:
        return self + other
    def __mul__(self: T, other: T) -> T: ...
    def __rmul__(self: T, other: T) -> T:
        return self * other
    def __sub__(self: T, other: T) -> T: ...
    def __rsub__(self: T, other: T) -> T: ...
    def __floordiv__(self: T, other: T) -> T: ...
    def __rfloordiv__(self: T, other: T) -> T:
        return type(self)(other) // self
    def __mod__(self: T, other: T) -> T: ...
    def __rmod__(self: T, other: T) -> T:
        return type(self)(other) % self
    def __pow__(self: T, other: int) -> T: ...
    def __eq__(self: T, other: T) -> bool: ...
    def __ne__(self: T, other: T) -> bool:
        return not self == other
    def __lt__(self: T, other: T) -> bool: ...
    def __le__(self: T, other: T) -> bool: ...
    def __gt__(self: T, other: T) -> bool: ...
    def __ge__(self: T, other: T) -> bool: ...
    def __hash__(self: T) -> int: ...
    def __neg__(self: T) -> T: ...
    def __repr__(self: T) -> str: ...



def isVal(e):
    return isinstance(e, int) or isinstance(e, Polynomial)


FQOrVal = Union[T_FQ, FQVal]




@total_ordering
class Polynomial(ABC, dict, FQVal):

    deg = None
    var = None
    coef_type = None


    def __init__(
            self: T_Polynomial,
            values: Union[Sequence[FQVal], Mapping[int, FQVal], FQVal] = 0,
            *posvalues: Union[Sequence[FQVal], Mapping[int, FQVal], FQVal],
            ) -> None:

        if len(posvalues) > 0:
            values = [values] + list(posvalues)

        if self.coef_type is None:
            raise AttributeError("Coefficient type not specified")

        if isinstance(values, Mapping):
            vals = []
            for k,v in values.items():
                c = self.coef_type(v)
                if c != 0:
                    vals.append((k,c))

        elif isinstance(values, Sequence):
            vals = []
            for k,v in enumerate(values):
                c = self.coef_type(v)
                if c != 0:
                    vals.append((k,c))
        else:
            c = self.coef_type(values)
            vals = [(0,c)] if c != 0 else []

        super().__init__(vals)

        self.deg = max(-1,-1,*(self.keys()))

    def __missing__(self, key):
        return 0

    def __setitem__(self, key, item):
        if item != 0:
            if key > self.deg:
                self.deg = key
            return super().__setitem__(key, item)
        elif key in self:
            super().__delitem__(key)
            self.deg = max(-1,-1,*(self.keys()))
    

    def __add__(self: T_Polynomial, other: FQVal) -> T_Polynomial:
        if isinstance(other, Polynomial):
            return type(self)({i: self[i] + other[i] 
                    for i in set(self).union(set(other))})
        else:
            new = type(self)(self)
            new[0] += other
            return new


    def __mul__(self: T_Polynomial, other: FQVal) -> T_Polynomial:
        if isinstance(other, Polynomial):
            newp = type(self)()
            for i,v in self.items():
                for j,w in other.items():
                    newp[i + j] += v * w
            return newp
        else:
            return type(self)({i: v * other for i,v in self.items()})
            

    def __sub__(self: T_Polynomial, other: FQVal) -> T_Polynomial:
        if isinstance(other, Polynomial):
            return type(self)({i: self[i] - other[i]
                    for i in set(self).union(other)})
        else:
            newp = type(self)(self)
            newp[0] -= other
            return newp

    def __rsub__(self: T_Polynomial, other: FQVal) -> T_Polynomial:
        if isinstance(other, Polynomial):
            return type(self)({i: other[i] - self[i]
                    for i in set(self).union(other)})
        else:
            newp = -self
            newp[0] += other
            return newp

    def __floordiv__(self: T_Polynomial, other: FQVal) -> T_Polynomial:
        if isinstance(other, Polynomial):
            quot, rem = polynomial_division(self, other)
            return quot
        else:
            return type(self)({i: v // other for i,v in self.items()})

    def __mod__(self: T_Polynomial, other: FQVal) -> T_Polynomial:
        if isinstance(other, Polynomial):
            quot, rem = polynomial_division(self, other)
            return rem
        else:
            return type(self)()

    def __pow__(self: T_Polynomial, other: FQVal) -> T_Polynomial:
        return fast_exponentiation(self, other)

    def __eq__(self: T_Polynomial, other: FQVal) -> bool:
        if isinstance(other, Polynomial):
            c1 = sorted((k,v) for k,v in self.items() if v != 0)
            c2 = sorted((k,v) for k,v in other.items() if v != 0)
            return self.var == other.var and tuple(c1) == tuple(c2)
        else:
            return self.deg <= 0 and self[0] == other

    def __ne__(self: T_Polynomial, other: FQVal) -> bool:
        return not (self == other)

    def __lt__(self: T_Polynomial, other: FQVal) -> bool:
        if isinstance(other, Polynomial):
            if self.deg < other.deg:
                return True
            elif self.deg > other.deg:
                return False
            else:
                return self[self.deg] < other[other.deg]
        else:
            return self.deg == 0 and self[0] < other

    def __hash__(self: T_Polynomial) -> int:
        return hash((self.var,tuple(sorted((k,v) for k,v in self.items() if v != 0))))

    def __neg__(self: T_Polynomial) -> T_Polynomial:
        return type(self)({i: -v for i,v in self.items()})

    def __repr__(self: T_Polynomial) -> str:
        if self == 0:
            return "0"
        return ' + '.join(reversed([f"{'' if v == 0 or (v == 1 and i > 0) else repr(v)}{self.var if i > 0 else ''}{print_superscript(i) if i > 1 else ''}" for i,v in sorted(list(self.items())) if v != 0]))


"""
    Factory method for polynomial rings
"""
def Polynomials(cls, var = "x"):
    return type(f"Polynomial{cls.__name__}", (Polynomial,), {'coef_type': cls, 'var': var})





"""
    Class of elements of a finite field
"""
@total_ordering
class FQ():

    val = None
    modulus = None

    def __init__(self: T_FQ, val: FQOrVal) -> None:
        if self.modulus is None:
            raise AttributeError("Missing modulus")

        if isinstance(val, type(self.modulus)):
            self.val = val % self.modulus
        elif isinstance(val, FQ):
            self.val = val.val % self.modulus
        else:
            self.val = type(self.modulus)(val) % self.modulus


    def __add__(self: T_FQ, other: FQOrVal) -> T_FQ:
        ot = other if isVal(other) else other.val
        return type(self)(self.val + ot)

    def __radd__(self: T_FQ, other: FQOrVal) -> T_FQ:
        return self + other

    def __mul__(self: T_FQ, other: FQOrVal) -> T_FQ:
        ot = other if isVal(other) else other.val
        return type(self)(self.val * ot)

    def __rmul__(self: T_FQ, other: FQOrVal) -> T_FQ:
        return self * other

    def __sub__(self: T_FQ, other: FQOrVal) -> T_FQ:
        ot = other if isVal(other) else other.val
        return type(self)(self.val - ot)

    def __rsub__(self: T_FQ, other: FQOrVal) -> T_FQ:
        ot = other if isVal(other) else other.val
        return type(self)(ot - self.val)

    def __truediv__(self: T_FQ, other: FQOrVal) -> T_FQ:
        ot = other if isVal(other) else other.val
        return type(self)(self.val * mod_inv(ot, self.modulus))

    def __rtruediv__(self: T_FQ, other: FQOrVal) -> T_FQ:
        ot = other if isVal(other) else other.val
        return type(self)(mod_inv(self.val, self.modulus) * ot)

    def inv(self: T_FQ) -> T_FQ:
        return type(self)(mod_inv(self.val, self.modulus))

    def __pow__(self: T_FQ, other: int) -> T_FQ:
        if other < 0:
            return self.inv() ** (-other)
        else:
            return fast_exponentiation(self, other)


    def __eq__(self: T_FQ, other: FQOrVal) -> bool:
        ot = other if isVal(other) else other.val
        return self.val == ot

    def __lt__(self: T_FQ, other: FQOrVal) -> bool:
        ot = other if isVal(other) else other.val
        return self.val < ot

    def __hash__(self: T_FQ) -> int:
        return hash(self.val)

    def __neg__(self: T_FQ) -> T_FQ:
        return type(self)(-self.val)

    def __repr__(self: T_FQ) -> str:
        return repr(self.val)


    @classmethod
    def randelem(cls):
        p = cls.char()
        q = cls.order()
        if q != p:
            coefs = [random.randint(0,p-1) for i in range(cls.modulus.deg)]
            rho = cls(coefs)
            if rho == 0:
                return cls.randelem()
        else:
            rho = cls(random.randint(1,q-1))
        return rho

    def is_quadratic(self) -> bool:
        return self**((self.order()-1)//2) == 1
        

    def sqrt(self: T_FQ) -> T_FQ:
        # Adleman-Manders-Miller
        p = self.char()
        q = self.order()

        rho = self.randelem()
        while rho.is_quadratic():
            rho = self.randelem()

        s = q-1
        t = 0
        while s % 2 == 0:
            s //= 2
            t += 1

        a = rho**s
        b = self**s
        h = 1
        for i in range(1,t):
            d = b**(2**(t-1-i))
            k = 0 if d == 1 else 1
            b = b*((a**2)**k)
            h = h*(a**k)
            a = a**2

        res = self**((s+1)//2)*h
        if res**2 != self:
            return None
        return sorted([res,-res])


    @classmethod
    def order(cls) -> int:
        if hasattr(cls.modulus, "__len__"):
            return cls.modulus.coef_type.order()**cls.modulus.deg
        else:
            return cls.modulus

    @classmethod
    def char(cls) -> int:
        if hasattr(cls.modulus, "__len__"):
            return cls.modulus.coef_type.char()
        else:
            return cls.modulus

        
    @classmethod
    def one(cls: type(T_FQ)) -> T_FQ:
        return cls(1)

    @classmethod
    def zero(cls: type(T_FQ)) -> T_FQ:
        return cls(0)

    @classmethod
    def gen(cls: type(T_FQ)) -> T_FQ:
        if isinstance(cls.modulus, Polynomial):
            return cls([0,1])
        else:
            return cls(2) if cls.modulus > 2 else cls(1)



"""
    Factory method for generating finite fields with
    a given moduls.
"""
def FQFac(mod):

    class meta(type):
        def __getitem__(cls, key):
            return Polynomials(cls, var=key)


    @classmethod
    def extend(cls, coefs: Union[Sequence[FQ], Mapping[int, FQ]], var = None):
        if var is not None:
            pol = Polynomials(cls, var)(coefs)
        else:
            pol = Polynomials(cls)(coefs)
        return FQFac(pol)

    return meta(f"FQ/{mod}", (FQ,), {'modulus': mod, 'extend': extend})



# Helper functions


def polynomial_division(pol: Polynomial, div: Polynomial):

    tpol = type(pol)

    if div == 0:
        raise ValueError("Can't divide polynomial by 0")

    quot = tpol()
    rem = tpol(pol)

    lcdiv = div[div.deg]
    T_c = type(lcdiv)

    while rem.deg >= div.deg:

        # Leading coefficient
        lc = rem[rem.deg]

        degdif = rem.deg - div.deg

        nquot = T_c(lc / lcdiv)

        for i,v in list(div.items()):
            rem[i+degdif] -= v*nquot

        quot[degdif] += nquot

   
    if pol.deg >= div.deg and (not rem.deg < div.deg):
        print(pol, div, quot, rem)
        raise ValueError("Attempted division with non-euclidean coefficients")

    
    return quot, rem
    


def mod_inv(val, modulus):
    # Extended euclidean algorithm
    r = [modulus, val]
    s = [1, 0]
    t = [0, 1]
    
    while r[-1] != 0:
        q = r[-2] // r[-1]
        r.append(r[-2] - q*r[-1])
        s.append(s[-2] - q*s[-1])
        t.append(t[-2] - q*t[-1])

    if not is_unit(r[-2]):
        raise ValueError(f"{val} has no inverse mod {modulus}")

    if r[-2] != 1:
        return t[-2] // r[-2]
    else:
        return t[-2]


def is_unit(x: FQVal):
    if isinstance(x, int):
        return x == 1 or x == -1
    else:
        return x.deg == 0 and x != 0


def fast_exponentiation(a: FQOrVal, n: int):
    """
        Computes a^n in O(log(n)) time
    """

    res = 1
    acc = a
    while n > 0:
        if n % 2 == 1:
            res *= acc
        n //= 2
        if n > 0:
            acc = acc*acc

    return res

def print_superscript(n: int):
    uni = [
        "\U00002070",
        "\U000000B9",
        "\U000000B2",
        "\U000000B3",
        "\U00002074",
        "\U00002075",
        "\U00002076",
        "\U00002077",
        "\U00002078",
        "\U00002079"
    ]
    digits = []
    while n > 0:
        digits.append(uni[n % 10])
        n //= 10

    return ''.join(digits[::-1])

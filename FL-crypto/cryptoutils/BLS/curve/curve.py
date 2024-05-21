from .curve_fields import FQ, FQ2, FQ12
from .fields import T_FQ
from collections import namedtuple


Point = namedtuple("Point", ['x', 'y'])

#Represents an eliptic curve point
class CurvePoint(Point):

    # Curve equation parameters: y^2 = x^3 + ax + b
    b = None
    a = None


    def __new__(cls, x = None, y = None):
        if cls.a is None or cls.b is None:
            raise AttributeError("Curve paremeters not set")
        if x is not None and y is None:
            raise NotImplementedError()
        self = super(CurvePoint, cls).__new__(cls, x, y)
        return self

    def __init__(self, x = None, y = None) -> None:
        if not self.is_on_curve():
            raise ValueError("Point is not in curve")

#Infinite for us is Null since we dont use projective coordinates 
    def is_infinite(self):
        return self.x is None

    def is_on_curve(self):
        x, y = self
        return self.is_infinite() or y**2 == x**3 + self.a * x + self.b


    def __add__(self, other):
        if self.is_infinite():
            return other
        if other.is_infinite():
            return self

        if self == other:
            return self.double()

        x1, y1 = self
        x2, y2 = other

        if x2 == x1:
            # self != other, so self = -other
            # return inf
            return type(self)()

        # Chord slope
        m = (y2 - y1) / (x2 - x1)

        newx = m**2 - x1 - x2
        newy = -(m * newx + y1 - m * x1)
        return type(self)(newx, newy)


    def __mul__(self, other: int):
        if self == G2:  # 针对G2做快速查表运算
            time = 0
            tmp = BLS12_381_FQ2()
            while other > 0:
                if other & 1:
                    tmp += G2_Table[time]
                other >>= 1
                time += 1
            return tmp
        else:
            if other == 0:
                return type(self)()
            elif other == 1:
                return self
            elif other % 2 == 0:
                return self.double() * (other // 2)
            else:
                return self.double() * (other // 2) + self

    def __rmul__(self, other: int):
        return self*other

#add a point to itself
    def double(self):
        if self.is_infinite():
            return self
        x, y = self
        m = 3 * x**2 / (2*y)
        newx = m**2 - 2*x
        newy = -(m * newx + y - m * x)
        return type(self)(newx, newy)

    def __neg__(self):
        x, y = self
        return type(self)(x, -y)

    def __eq__(self, other):
        if self.is_infinite() or other.is_infinite():
            return self.is_infinite() and other.is_infinite()
        else:
            return super().__eq__(other)

        #some curves use it, others dont
    #@abstractmethod
    def twist(self):
        raise NotImplementedError()



class BLS12_381_FQ(CurvePoint):

    a = 0
    b = FQ(4)

G1 = BLS12_381_FQ(
    FQ(3685416753713387016781088315183077757961620795782546409894578378688607592378376318836054947676345821548104185464507),
    FQ(1339506544944476473020471379941921221584933875938349620426543736416511423956333506472724655353366534992391756441569)
    )

curve_order = 52435875175126190479447740508185965837690552500527637822603658699938581184513


class BLS12_381_FQ12(CurvePoint):

    a = 0
    b = FQ12(4)

class BLS12_381_FQ2(CurvePoint):

    a = 0
    b = FQ2((4,4))

    w = FQ12.gen()

    def twist(self):
        if self.is_infinite():
            return self

        x, y = self

        # Base change inside FQ2
        xc = [x.val[0] - x.val[1], x.val[1]]
        yc = [y.val[0] - y.val[1], y.val[1]]

        # Embed into FQ12
        nx = FQ12({0:xc[0], 6:xc[1]})
        ny = FQ12({0:yc[0], 6:yc[1]})

        # Apply twist
        return BLS12_381_FQ12(nx / self.w**2, ny / self.w**3)


G2 = BLS12_381_FQ2(
    FQ2([
        352701069587466618187139116011060144890029952792775240219908644239793785735715026873347600343865175952761926303160,
        3059144344244213709971259814753781636986470325476647558659373206291635324768958432433509563104347017837885763365758]),
    FQ2([
        1985150602287291935568054521177171638300868978215655730859378665066344726373823718423869104263333984641494340347905,
        927553665492332455747201965776037880757740193453592970025027978793976877002675564980949289727957565575433344219582])
    )

G12 = G2.twist()

G2_Table = [G2]
def makeTable():
    tmp = G2
    for i in range(400):
        tmp = tmp.double()
        G2_Table.append(tmp)


makeTable()

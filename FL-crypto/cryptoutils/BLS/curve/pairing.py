from .curve import Point, CurvePoint, G1, curve_order
from .curve_fields import FQ, FQ2, FQ12, BLS12_381_FQ_MODULUS as field_modulus
import time

# number of iterations that Miller's loop algorithm requires
ate_loop_count = 15132376222941642752
log_ate_loop_count = 62


# returns the line between P1 and P2 evaluated in Q
def linefunc(P1: CurvePoint, P2: CurvePoint, Q: Point) -> FQ:
    if P1.is_infinite() or P2.is_infinite() or Q.is_infinite():
        raise ValueError("Can't compute line function on infinite point")

    x1, y1 = P1
    x2, y2 = P2
    xq, yq = Q

    if x1 != x2:
        m = (y2 - y1) / (x2 - x1)
        return m * (xq - x1) - (yq - y1)
    elif y1 == y2:
        m = 3 * x1 ** 2 / (2 * y1)
        return m * (xq - x1) - (yq - y1)
    else:
        return xq - x1


# embeds a FQ point into a FQ12 point
def embed_FQ12(P: Point) -> Point:
    x, y = P
    return type(P)(FQ12(x), FQ12(y))


# algorithm which calculates pairing
def miller_loop(P: CurvePoint, Q: CurvePoint) -> FQ12:
    if P.is_infinite() or Q.is_infinite():
        return FQ12.one()

    R = Q
    f = FQ12.one()
    num = 0
    for i in range(log_ate_loop_count, -1, -1):
        num += 1
        f = f * f * linefunc(R, R, P)
        R = 2 * R
        if ate_loop_count & (2 ** i):
            f = f * linefunc(R, Q, P)
            R = R + Q
    # ans = f ** ((field_modulus ** 12 - 1) // curve_order)
    # ans = fast_pow(f)
    ans = fast_pow_2(f)
    return ans


def pairing(P: CurvePoint, Q: CurvePoint) -> FQ12:
    return miller_loop(embed_FQ12(P), Q.twist())


def fast_pow(f: FQ12):
    # t1 = f ** (field_modulus ** 6)
    t1 = fast_pow_fp(fast_pow_fp(fast_pow_fp(fast_pow_fp(fast_pow_fp(fast_pow_fp(f))))))
    t2 = t1 * f.inv()
    t3 = fast_pow_fp(fast_pow_fp(t2))
    t4 = t3 * t2
    # print(t4 == f ** ((field_modulus**6-1)*(field_modulus**2+1)))
    u = -0xd201000000010000
    m = [t4]*10
    m[1] = t4 ** (-2)
    m[2] = t4 ** u
    m[3] = m[2] ** 2
    m[4] = m[1] * m[2]
    m[5] = m[4] ** u
    m[6] = m[5] ** u
    m[7] = m[6] ** u
    m[8] = m[3] * m[7]
    m[9] = m[8] ** u
    final_ans = m[9]*m[4].inv()*t4*fast_pow_fp(m[8]*t4.inv())*fast_pow_fp(fast_pow_fp(m[6]*m[2]))*fast_pow_fp(fast_pow_fp(fast_pow_fp(m[5]*t4)))
    return final_ans


def fast_pow_2(f:FQ12):
    t1 = fast_pow_fp(fast_pow_fp(fast_pow_fp(fast_pow_fp(fast_pow_fp(fast_pow_fp(f))))))
    t2 = t1 * f.inv()
    t3 = fast_pow_fp(fast_pow_fp(t2))
    t4 = t3 * t2
    u = -0xd201000000010000
    t5 = (t4 ** u) * t4.inv()
    t6 = (t5 ** u) * t5.inv()
    t7 = (t6 ** u) * fast_pow_fp(t6)
    t8 = (t7 ** u) ** u * fast_pow_fp(fast_pow_fp(t7)) * t7.inv()
    t9 = t8 * (t4 ** 3)

    ans = t9
    return ans


def fast_pow_fp(f: FQ12):
    ans = FQ12(0)
    for i in range(12):
        ans += FQ12(f.val[i]) * list_for_u[i]
    return ans


def make_u_list():
    us = [FQ12((1)), FQ12((0, 1)), FQ12((0, 0, 1)), FQ12((0, 0, 0, 1)), FQ12((0, 0, 0, 0, 1)), FQ12((0, 0, 0, 0, 0, 1)),
          FQ12((0, 0, 0, 0, 0, 0, 1)), FQ12((0, 0, 0, 0, 0, 0, 0, 1)), FQ12((0, 0, 0, 0, 0, 0, 0, 0, 1)),
          FQ12((0, 0, 0, 0, 0, 0, 0, 0, 0, 1)), FQ12((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)),
          FQ12((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1))]
    for u in us:
        list_for_u.append(u ** field_modulus)


list_for_u = []
make_u_list()
